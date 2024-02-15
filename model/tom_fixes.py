import numpy as np
import onnx

from onnx import helper
import matplotlib.pyplot as plt
import onnxruntime as rt


def add_unsqueeze(graph, output_name, axis=0):
    # Create a new unsqueeze node
    unsqueeze_node = helper.make_node(
        'Unsqueeze',
        inputs=[output_name],
        outputs=[output_name + '_unsqueezed'],
        axes=[axis]
    )

    # Add the new unsqueeze node to the graph
    graph.node.extend([unsqueeze_node])

    # Update the output name to the unsqueezed one
    output_name = output_name + '_unsqueezed'

    return output_name


def plot_images(images, rows, cols):
    fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(images[i * cols + j, 0, :, :], cmap='gray')
            axs[i, j].axis('off')

    plt.show()


def find_best_matches(A, B):
    n = A.shape[0]
    best_matches = np.ones(n, dtype=int) * (-1)
    for i in range(n):
        best_distance = float('inf')
        best_index = -1
        for j in range(n):
            if j not in best_matches:
                distance = np.sum((A[i] - B[j]) ** 2)
                if distance < best_distance:
                    best_distance = distance
                    best_index = j
        if best_index == -1:
            raise Exception("Could not greedily match the order of the two arrays")
        best_matches[i] = best_index
    return best_matches


def efficiency_test():
    onnx_model_path = f'modified_model.onnx'
    # Load your ONNX model
    model = onnx.load(onnx_model_path)
    sess = rt.InferenceSession(onnx_model_path)
    output_tensor_names = []
    output_shapes = []
    for output in sess.get_outputs():
        if output.shape[0] != 1 and output.name != 'anchors':
            output_tensor_names.append(output.name)
            output_shapes.append(output.shape)
    # Assuming you have a list of N output tensor names
    # output_tensor_names = ['output1', 'output2', 'output3']  # Replace with your actual tensor names
    output_indices = []
    for output_tensor in output_tensor_names:
        for i, output in enumerate(model.graph.output):
            if output.name == output_tensor:
                output_indices.append(i)
    # Iterate through each output tensor
    for i, output_name in enumerate(output_tensor_names):
        # Find the output node in the graph
        # output_node = next(node for node in model.graph.node if output_name in node.output)
        # Add an Unsqueeze node to the output tensor
        new_output_name = add_unsqueeze(model.graph, output_name)

        # Add the new output to the graph
        model.graph.output.insert(output_indices[i] + 1, helper.make_tensor_value_info(new_output_name,
                                                                                       model.graph.output[
                                                                                           output_indices[
                                                                                               i]].type.tensor_type.elem_type,
                                                                                       [1, *output_shapes[i]]))
        model.graph.output.remove(model.graph.output[output_indices[i]])
        # Remove the previous output from the graph
    # for i, output in enumerate(model.graph.output):
    #     if output.name == 'anchors':
    #         model.graph.output.remove(output)
    # for idx in output_indices[::-1]:
    #     model.graph.output.remove(model.graph.output[idx])
    predictor_conv_idx = \
    [i for i in range(len(model.graph.node)) if '/roi_heads/mask_head/predictor' in model.graph.node[i].name][0]
    predictor_out = model.graph.node[predictor_conv_idx].output[0]
    conv_bias = model.graph.node[predictor_conv_idx].input[2]
    # Create the addition node
    result_name = 'mask_loss_features_fix'
    addition_node = helper.make_node(
        'Add',
        inputs=[predictor_out, conv_bias],
        outputs=[result_name],
        name='addition_node'
    )
    model.graph.node[predictor_conv_idx].input.remove(
        model.graph.node[predictor_conv_idx].input[2])  # remove bias from conv
    #
    # replace inputs
    for j in range(len(model.graph.node)):
        for i, inpt in enumerate(model.graph.node[j].input):
            if predictor_out == model.graph.node[j].input[i]:
                model.graph.node[j].input[i] = result_name
    model.graph.node.insert(predictor_conv_idx + 1, addition_node)
    onnx_path = 'final_model_11022024.onnx'
    onnx.save(model, onnx_path)
    sess = rt.InferenceSession(onnx_path)

    # Save the modified m


if __name__ == '__main__':
    efficiency_test()

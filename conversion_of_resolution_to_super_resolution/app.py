import torch
import qai_hub as hub
from qai_hub_models.models.esrgan import Model
from PIL import Image
import torchvision.transforms as transforms
import os

def enhance_image(image_path, output_path='high_resolution_image.jpg', device_name="Samsung Galaxy S24"):
    """
    Enhance a blurry image using ESRGAN on a specific device.
    
    Args:
        image_path (str): Path to the input blurry image
        output_path (str): Path to save the enhanced image
        device_name (str): Name of the device to run inference on
    """
    print(f"Loading image from: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Load the model
    print("Loading ESRGAN model...")
    torch_model = Model.from_pretrained()
    
    # Set device
    device = hub.Device(device_name)
    print(f"Using device: {device_name}")
    
    # Get input specification and sample inputs
    input_spec = torch_model.get_input_spec()
    sample_inputs = torch_model.sample_inputs()
    
    # Trace model
    print("Tracing PyTorch model...")
    pt_model = torch.jit.trace(torch_model, [torch.tensor(data[0]) for _, data in sample_inputs.items()])
    
    # Compile model on the target device
    print(f"Compiling model for {device_name}...")
    compile_job = hub.submit_compile_job(
        model=pt_model,
        device=device,
        input_specs=input_spec,
    )
    
    # Wait for compilation to complete
    compile_job.wait()
    print("Compilation complete")
    
    # Get the target model
    target_model = compile_job.get_target_model()
    
    # Load and preprocess the image
    print("Processing input image...")
    image = Image.open(image_path).convert('RGB')
    
    # Get input dimensions from the input spec
    input_name = list(input_spec.keys())[0]
    input_shape = input_spec[input_name][0]  # Shape: (batch_size, channels, height, width)
    _, _, height, width = input_shape
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Run inference on the device
    print("Running inference...")
    inference_job = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs={input_name: image_tensor.numpy()},
    )
    
    # Wait for the inference to complete
    inference_job.wait()
    print("Inference complete")
    
    # Get and process the output
    output_data = inference_job.get_output()
    output_tensor = torch.tensor(output_data[list(output_data.keys())[0]])
    
    # Convert output tensor to image
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    
    # Save the enhanced image
    output_image.save(output_path)
    print(f"Enhanced image saved as '{output_path}'")
    
    return output_path

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance a blurry image using ESRGAN')
    parser.add_argument('--input', type=str, required=True, help='Path to the input blurry image')
    parser.add_argument('--output', type=str, default='high_resolution_image.jpg', help='Path to save the enhanced image')
    parser.add_argument('--device', type=str, default='Samsung Galaxy S24', help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Install required packages if not already installed
    try:
        import importlib
        required_packages = ['torch', 'torchvision', 'pillow', 'qai_hub', 'qai_hub_models']
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                print(f"Installing {package}...")
                import subprocess
                subprocess.check_call(['pip', 'install', package])
    except Exception as e:
        print(f"Error installing packages: {e}")
        print("Please manually install required packages: pip install torch torchvision pillow qai_hub qai_hub_models")
    
    enhance_image(args.input, args.output, args.device)
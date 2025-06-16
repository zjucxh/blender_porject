import blenderproc as bproc
import numpy as np

if __name__ == "__main__":
    # Initialize BlenderProc
    bproc.init()

    # Create monkey object
    monkey = bproc.object.create_primitive("MONKEY")
    
    # Set light
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([2, -2, 0])
    light.set_energy(500)

    # Set the camera to be in front of the object
    cam_pose = bproc.math.build_transformation_mat([0, -5, 0], [np.pi / 2, 0, 0])
    bproc.camera.add_camera_pose(cam_pose)

    # Render the scene
    bproc.renderer.set_output_format("PNG")
    bproc.renderer.set_resolution(640, 480)
    bproc.renderer.render()
    # Save the rendered image
    bproc.writer.write_blend("output/output.blend")
    bproc.writer.write_images("output/output_images")
    bproc.writer.write_depth("output/output_depth")
    bproc.writer.write_camera_poses("output/output_camera_poses.json")
    bproc.writer.write_object_instances("output/output_object_instances.json")
    # Save the scene
    bproc.writer.write_scene("output/output_scene.blend")
    # Print the output path
    print("Rendered scene saved to 'output/output_scene.blend' and images saved to 'output_images/'")
    
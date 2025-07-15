from pyorbbecsdk import *

class CameraInterface:
    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()

    def setup_streams(self):
        # COLOR STREAM (force RGB format)
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_profiles.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError:
            color_profile = color_profiles.get_default_video_stream_profile()
        self.config.enable_stream(color_profile)
        print("Selected color format:", color_profile.get_format())


        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        self.config.enable_stream(depth_profile)

        self.pipeline.start(self.config)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(100)
        if frames:
            return frames.get_color_frame(), frames.get_depth_frame()
        return None, None

    def stop(self):
        self.pipeline.stop()

# from ultralytics import SAM2
#
# from get_foreground_people import GetForegroundPersons
#
#
# class SegmentPersons():
#     def __init__(self, source=0):
#         self.FastSAM_model = FastSAM("FastSAM-s.pt")
#         self.get_foreground = GetForegroundPersons(source='WIN_20250222_07_22_43_Pro.mp4')
#
#     def getForeGroundPersons(self):
#         poses = self.get_foreground.extract_last_pose()
#         return poses
#     def segment(self, poses):
#         for pose in poses:
#             self.FastSAM_model


from ultralytics import SAM

# Load a model
model = SAM("sam2.1_t.pt")
results = model('WIN_20250222_07_22_43_Pro.mp4', save=True)

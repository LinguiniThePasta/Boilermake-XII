''' FILE FOR CROSS SCRIPT VARIABLES???'''

detection_events = None
def get_detection_events():
    return detection_events

def set_detection_events(new_detection_events):
    global detection_events
    print(new_detection_events)
    detection_events = new_detection_events
    print(detection_events)
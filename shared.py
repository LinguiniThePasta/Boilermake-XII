''' FILE FOR CROSS SCRIPT VARIABLES???'''

detection_events = None
huge_shit = {}

def get_huge_shit(key):
    return huge_shit[key]

def add_huge_shit(key, value):
    global huge_shit
    if key in huge_shit:
        huge_shit[key].append(value)
    else:
        huge_shit[key] = value

    print(huge_shit[key])

def get_detection_events():
    return detection_events

def set_detection_events(new_detection_events):
    global detection_events
    print(new_detection_events)
    detection_events = new_detection_events
    print(detection_events)
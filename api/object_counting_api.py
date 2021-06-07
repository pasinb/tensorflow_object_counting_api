#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util
import multiprocessing

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print("CV version: " + cv2.__version__)

# TODO
# research new model
# rate limit tracker / obj detect 
# improve intersection
# improve count detection (2 lines?, tracker in out position)
# tracker ID

def tracker_process(input_queue, output_queue):
    trackers = {}
    while True:
        data = input_queue.get()
        if data is not None:
            if data['type'] == 'create_tracker':
                tracker = cv2.TrackerKCF_create()
                tracker.init(data['frame'], data['init_bbox'])
                trackers[data['id']] = { 'tracker': tracker }
            elif data['type'] == 'get_tracker_count':
                output_queue.put({'tracker_count': len(trackers)})
            elif data['type'] == 'update_tracker':
                output = {'type': 'update_tracker', 'data': {}}
                for tracker_id, tracker in trackers.items():
                    output['data'][tracker_id] = {}
                    track_ok, track_bbox = trackers[tracker_id]['tracker'].update(data['frame'])
                    if track_ok:
                        # Tracking success
                        output['data'][tracker_id]['track_ok'] = True
                        output['data'][tracker_id]['updated_bbox'] = [int(v) for v in track_bbox]
                    else:
                        output['data'][tracker_id]['track_ok'] = False
                output_queue.put(output)
            elif data['type'] == 'remove_tracker':
                if data['id'] in trackers:
                    del trackers[data['id']]
            else:
                raise Exception('unknown data from parent process')

def cumulative_object_counting_x_axis_webcam(detection_graph, category_index, is_color_recognition_enabled, roi, deviation, custom_object_name):
        OBJECT_DETECT_DELAY = 0.5
        OBJECT_DETECT_CONFIDENCE_THRESHOLD = 0.7
        TRACKER_UPDATE_DELAY = 0
        TRACKER_FAIL_COUNT_THRESHOLD = 20
        POOL_COUNT = multiprocessing.cpu_count() - 1
        COUNT_RIGHT = True
        INTERSECT_DELAY = 0.5

        total_passed_objects = 0
        vacant_tracker_id = 1

        # id (number): {bbox, last bbox, fail_count}
        tracker_data_list = {}

        last_detect_tick_count = None
        last_tracker_update_tick_count = None

        fps = 0

        input_queues = [multiprocessing.Queue() for x in range(POOL_COUNT) ]
        output_queues = [multiprocessing.Queue() for x in range(POOL_COUNT) ] 
        for i in range(POOL_COUNT):
            p = multiprocessing.Process(target=tracker_process, args=(input_queues[i], output_queues[i], ))
            p.daemon = True
            p.start()

        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # cap = cv2.VideoCapture(0)
            cap = cv2.VideoCapture(1)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # for all the frames that are extracted from input video
            while True:
                tick_count = cv2.getTickCount()
                tick_freq = cv2.getTickFrequency()

                ret, frame = cap.read()                
                if frame is None:
                    continue
                if not ret:
                    print("end of the video file...")
                    break

                rows = frame.shape[0]
                cols = frame.shape[1]
                cv2.line(frame, (int(cols/2), 0), (int(cols/2), rows), (0, 0, 255), thickness=2)

                # Update tracker
                if last_tracker_update_tick_count == None or (tick_count - last_tracker_update_tick_count) / tick_freq > TRACKER_UPDATE_DELAY:
                    last_tracker_update_tick_count = tick_count

                    for i in range(len(input_queues)):
                        input_queues[i].put({'type':'update_tracker', 'frame': frame})

                    for i in range(len(output_queues)):
                        # TODO timeout?

                        # id, bbox, last_bbox, fail_count
                        res = output_queues[i].get()
                        if 'type' in res and res['type'] == 'update_tracker':
                            for key_id, track_result in res['data'].items():
                                if key_id in tracker_data_list:
                                    if track_result['track_ok'] is True:
                                        tracker_data_list[key_id]['fail_count'] = 0
                                        tracker_data_list[key_id]['last_bbox'] = tracker_data_list[key_id]['bbox'] 
                                        tracker_data_list[key_id]['bbox'] = track_result['updated_bbox']
                                    else: 
                                        tracker_data_list[key_id]['fail_count'] = tracker_data_list[key_id]['fail_count'] + 1
                                        
                                else:
                                    raise Exception('Subprocess have tracking ID not tracked by main process')
                        else:
                            raise Exception('Subprocess wrong answer')

                # Remove failed tracker
                for tracker_id in list(tracker_data_list):
                # for i in reversed(range(len(tracker_data_list))):
                    if tracker_data_list[tracker_id]['fail_count'] > TRACKER_FAIL_COUNT_THRESHOLD:
                        for iq in input_queues:
                            iq.put({'type': 'remove_tracker', 'id': tracker_id})
                        del tracker_data_list[tracker_id]


                # Detect if tracker intersect center of screen
                    

                # Render tracker
                for tracker_id, tracker_data in tracker_data_list.items():
                    (x, y, w, h) = tracker_data['bbox']
                    if tracker_data['fail_count'] > 0:
                        color = (0,0,255)
                    else:
                        color = (0,255,0)

                        # detect if track box intersect center of screen
                        (prev_x, prev_y, prev_w, prev_h) = tracker_data['last_bbox']
                        center_x = cols/2
                        intersects_center = center_x > x and center_x < x + w
                        prev_intersects_center = center_x > prev_x and center_x < prev_x + prev_w
                        in_intersect_delay = (tick_count - tracker_data['last_intersect_tick_count']) / tick_freq < INTERSECT_DELAY
                        
                        if intersects_center:
                            tracker_data['last_intersect_tick_count'] = tick_count
                            if not in_intersect_delay and not prev_intersects_center:
                                if (x + w/2) > (prev_x + prev_w/2):
                                    total_passed_objects += 1 if COUNT_RIGHT else -1
                                else:
                                    total_passed_objects += -1 if COUNT_RIGHT else 1
                        
                        if in_intersect_delay:
                            color = (255,0,0)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
                    cv2.putText(frame, str(tracker_id), (int(x + 5) , int(y + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    

                # Tensorflow object detection
                if last_detect_tick_count == None or (tick_count - last_detect_tick_count) / tick_freq > OBJECT_DETECT_DELAY:
                    last_detect_tick_count = tick_count

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(frame, axis=0)

                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    num_detection = int(num[0])
                    for i in range(num_detection):
                        name = category_index[classes[0][i]]['name']
                        score = float(scores[0][i])
                        bbox = [float(v) for v in boxes[0][i]]
                        if score > OBJECT_DETECT_CONFIDENCE_THRESHOLD:
                            x = int(bbox[1] * cols)
                            y = int(bbox[0] * rows)
                            right = int(bbox[3] * cols)
                            bottom = int(bbox[2] * rows)
                            # area = round((right - x) * (bottom - y) / 1000)
                            # aspect_ratio = round((right - x) / (bottom - y), 2)
                            cv2.rectangle(frame, (x, y), (right, bottom), (0, 255, 255), thickness=2)
                            cv2.putText(frame, name + ' ' + str(round(score * 100)) + '%' , (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            # cv2.putText(frame,  str(area) + ' / ' + str(aspect_ratio) , (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            if name == 'person' :
                                center_x = int((x + right) / 2)
                                center_y = int((y + bottom) / 2)

                                # check if there's already a tracker here
                                intersect_existing_trackeers = False
                                for tracker_id, tracker_data in tracker_data_list.items():
                                    (x_t, y_t, w_t, h_t) = tracker_data['bbox']
                                    if center_x > x_t and center_x < x_t + w_t and center_y > y_t and center_y < y_t + h_t:
                                        intersect_existing_trackeers = True
                                        break

                                # add new tracker for person
                                if not intersect_existing_trackeers:
                                    track_bbox = (x, y, right - x, bottom - y)

                                    # find least busy process
                                    for iq in input_queues:
                                        iq.put({'type': 'get_tracker_count'})
                                    process_data = []
                                    for j in range(len(output_queues)):
                                        d = output_queues[j].get()
                                        if 'tracker_count' in d:
                                            process_data.append({'idx': j, 'tracker_count': d['tracker_count']})
                                        else:
                                            raise Exception('Subprocess wrong answer')
                                    process_data = sorted(process_data, key=lambda k: k['tracker_count']) 
                                    least_busy_process_idx = process_data[0]['idx']
                                    
                                    # find vacant tracking ID
                                    # vacant_tracker_id = vacant_tracker_id + 1
                                    vacant_tracker_id = 0
                                    while True:
                                        if vacant_tracker_id in tracker_data_list:
                                            vacant_tracker_id += 1
                                            continue
                                        else:
                                            break

                                    # add tracking data to main process and subprocess
                                    tracker_data_list[vacant_tracker_id] = {'bbox': track_bbox, 'last_bbox': track_bbox, 'fail_count': 0, 'last_intersect_tick_count': 0}
                                    input_queues[least_busy_process_idx].put({'type': 'create_tracker', 'frame': frame, 'init_bbox': track_bbox, 'id': vacant_tracker_id })

                # show count
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'Count: ' + str(total_passed_objects) + ' Track: ' + str(len(tracker_data_list)),(10, 40),font, 1.5,(0, 255, 255),2)

                # Calculate and display FPS
                cv2.putText(frame, "FPS : " + str(int(fps)), (10, 60), font, 0.5, (0, 255, 255), 1);
                cv2.imshow('object counting', cv2.resize(frame, None, None, fx=2, fy=2))

                fps = tick_freq / (cv2.getTickCount() - tick_count);

                if cv2.waitKey(1) & 0xFF == ord('r'):
                    total_passed_objects = 0

            pool.terminate()
            cap.release()
            cv2.destroyAllWindows()
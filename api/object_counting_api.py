#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print("CV version: " + cv2.__version__)

# TODO
# rate limit tracker / obj detect 
# improve intersection
# improve count detection (2 lines?, nultiple ways?)

def cumulative_object_counting_x_axis_webcam(detection_graph, category_index, is_color_recognition_enabled, roi, deviation, custom_object_name):
        
        total_passed_objects = 0              

        # input video
        # cap = cv2.VideoCapture(input_video)


        # fps = int(cap.get(cv2.CAP_PROP_FPS))

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        total_passed_objects = 0
        # color = "waiting..."
        tracker_id = 1
        # multi_tracker = cv2.MultiTracker_create()
        trackers = []
        # tracker_initialized = False
        # track_bbox = None
        # track_ok = None

        # if int(minor_ver) < 3:
        # tracker = cv2.Tracker_create('KCF')
        # else:
        #     tracker = cv2.TrackerKCF_create()

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

            cap = cv2.VideoCapture(0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # for all the frames that are extracted from input video
            while True:
                ret, frame = cap.read()                
                if frame is None:
                    continue
                if not ret:
                    print("end of the video file...")
                    break
                
                # Start timer
                timer = cv2.getTickCount()
                
                # Update tracker

                # if tracker_initialized:
                for i in reversed(range(len(trackers))):
                    # [tracker, tracker_id, bbox_tuple]
                    track_ok, track_bbox = trackers[i][0].update(frame)

                    if track_ok:
                        # for i, track_bbox in enumerate(track_bboxes):
                        #     p1 = (int(newbox[0]), int(newbox[1]))
                        #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                        #     cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
                        # Tracking success
                        (prev_x, prev_y, prev_w, prev_h) = trackers[i][2]
                        (x, y, w, h) = [int(v) for v in track_bbox]

                        if abs((x + w/2) - cols/2) < 10 and abs((prev_x + prev_w/2) - cols/2) > 10:
                            if (prev_x + prev_w/2) - cols/2 > 0:
                                total_passed_objects  = total_passed_objects + 1
                            else:
                                total_passed_objects  = total_passed_objects - 1


                        trackers[i][2] = (x, y, w, h)

                        # print(track_bbox)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), thickness=2)
                        cv2.putText(frame, str(trackers[i][1]), (int(x + w/2) , int(y + h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    else:
                        trackers.pop(i)
                        # Tracking failure
                        # cv2.putText(frame, "Tracking fail", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                        # tracker_initialized = False
                        


                # if tracker_initialized is False:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # print(num)
                # print(classes)
                # print(scores)
                # print(boxes)

                # cv2.rectangle(frame, (100 ,150), (200, 250), (255,255,255), thickness=2) # test intersect

                rows = frame.shape[0]
                cols = frame.shape[1]
                cv2.line(frame, (int(cols/2), 0), (int(cols/2), rows), (0, 0, 255), thickness=2)

                num_detection = int(num[0])
                # print(type(num_detection))
                for i in range(num_detection):
                    # pass
                    # classId = int(classes[i])
                    name = category_index[classes[0][i]]['name']
                    score = float(scores[0][i])
                    bbox = [float(v) for v in boxes[0][i]]
                    if score > 0.5:
                        x = int(bbox[1] * cols)
                        y = int(bbox[0] * rows)
                        right = int(bbox[3] * cols)
                        bottom = int(bbox[2] * rows)
                        # print([x,y,right,bottom])
                        area = round((right - x) * (bottom - y) / 1000)
                        aspect_ratio = round((right - x) / (bottom - y), 2)



                        cv2.rectangle(frame, (x, y), (right, bottom), (0, 255, 255), thickness=2)
                        cv2.putText(frame, name + ' ' + str(round(score * 100)) + '%' , (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.putText(frame,  str(area) + ' / ' + str(aspect_ratio) , (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        # if name == 'person' and abs(aspect_ratio - 0.5) < 0.1 and abs(area - 25) < 5:
                        if name == 'person' :
                            # print(x,y,right,bottom)

                            # intersect_c = (0, 255, 255)
                            # not (self.top_right.x < other.bottom_left.x or self.bottom_left.x > other.top_right.x or self.top_right.y < other.bottom_left.y or self.bottom_left.y > other.top_right.y)
                            center_x = int((x + right) / 2)
                            center_y = int((y + bottom) / 2)

                            intersect_existing_trackeers = False
                            for j in range(len(trackers)):
                                (x_t, y_t, w_t, h_t) = trackers[j][2]
                                if center_x > x_t and center_x < x_t + w_t and center_y > y_t and center_y < y_t + h_t:
                                    intersect_existing_trackeers = True
                                    break

                            # def intersects(self, other):
                            #     return not (self.top_right.x < other.bottom_left.x or self.bottom_left.x > other.top_right.x or self.top_right.y < other.bottom_left.y or self.bottom_left.y > other.top_right.y)
                            #     return not (self.top_right.x < other.bottom_left.x or self.bottom_left.x > other.top_right.x or self.top_right.y < other.bottom_left.y or self.bottom_left.y > other.top_right.y)
                            if not intersect_existing_trackeers:
                                # print('init tracker')
                                track_bbox = (x, y, right - x, bottom - y)
                                # track_ok = tracker.init(frame, track_bbox)
                                tracker = cv2.TrackerKCF_create()
                                tracker.init(frame, track_bbox)
                                trackers.append([tracker, tracker_id, [int(v) for v in track_bbox] ])
                                tracker_id = tracker_id + 1
                                # multi_tracker.add( cv2.Tracker_create('KCF'), frame, track_bbox)
                                # tracker_initialized = True

                # Visualization of the results of a detection.        
                # counter, csv_line, counting_result = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(cap.get(1),
                #                                                                                              frame,
                #                                                                                              is_color_recognition_enabled,
                #                                                                                              np.squeeze(boxes),
                #                                                                                              np.squeeze(classes).astype(np.int32),
                #                                                                                              np.squeeze(scores),
                #                                                                                              category_index,
                #                                                                                              x_reference = roi,
                #                                                                                              deviation = deviation,
                #                                                                                              use_normalized_coordinates=True,
                #                                                                                              line_thickness=4)
                               
                # # when the object passed over line and counted, make the color of ROI line green
                # if counter == 1:
                #   cv2.line(frame, (roi, 0), (roi, height), (0, 0xFF, 0), 5)
                # else:
                #   cv2.line(frame, (roi, 0), (roi, height), (0, 0, 0xFF), 5)

                # total_passed_objects = total_passed_objects + counter

                # output_movie.write(frame)
                # print ("writing frame")

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'Count: ' + str(total_passed_objects) + ' Track: ' + str(len(trackers)),(10, 15),font, 0.5,(0, 255, 255),1)

                # cv2.putText(frame,'ROI Line',(545, roi-10),font,0.6,(0, 0, 0xFF),2,cv2.LINE_AA,)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(fps)), (10, 30), font, 0.5, (0, 255, 255), 1);
    
                cv2.imshow('object counting', cv2.resize(frame, None, None, fx=2, fy=2))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

def cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation, custom_object_name):
        total_passed_objects = 0              

        # input video
        cap = cv2.VideoCapture(input_video)

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        total_passed_objects = 0
        color = "waiting..."
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

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, counting_result = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(boxes),
                                                                                                             np.squeeze(classes).astype(np.int32),
                                                                                                             np.squeeze(scores),
                                                                                                             category_index,
                                                                                                             x_reference = roi,
                                                                                                             deviation = deviation,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)
                               
                # when the object passed over line and counted, make the color of ROI line green
                if counter == 1:
                  cv2.line(input_frame, (roi, 0), (roi, height), (0, 0xFF, 0), 5)
                else:
                  cv2.line(input_frame, (roi, 0), (roi, height), (0, 0, 0xFF), 5)

                total_passed_objects = total_passed_objects + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected ' + custom_object_name + ': ' + str(total_passed_objects),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi-10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )

                output_movie.write(input_frame)
                print ("writing frame")
                #cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

def cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation, custom_object_name, targeted_objects=None):
        total_passed_objects = 0        

        # input video
        cap = cv2.VideoCapture(input_video)

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        total_passed_objects = 0
        color = "waiting..."
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

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

               # Visualization of the results of a detection.        
                counter, csv_line, counting_result = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(boxes),
                                                                                                             np.squeeze(classes).astype(np.int32),
                                                                                                             np.squeeze(scores),
                                                                                                             category_index,
                                                                                                             targeted_objects = targeted_objects,
                                                                                                             y_reference = roi,
                                                                                                             deviation = deviation,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)

                # when the object passed over line and counted, make the color of ROI line green
                if counter == 1:                  
                  cv2.line(input_frame, (0, roi), (width, roi), (0, 0xFF, 0), 5)
                else:
                  cv2.line(input_frame, (0, roi), (width, roi), (0, 0, 0xFF), 5)
                
                total_passed_objects = total_passed_objects + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected ' + custom_object_name + ': ' + str(total_passed_objects),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )               
                
                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi-10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )

                output_movie.write(input_frame)
                print ("writing frame")
                #cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()


def object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled):

        # input video
        cap = cv2.VideoCapture(input_video)

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        total_passed_objects = 0
        color = "waiting..."
        height = 0
        width = 0
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

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, counting_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if(len(counting_result) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    cv2.putText(input_frame, counting_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                output_movie.write(input_frame)
                print ("writing frame")
                #cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

def object_counting_webcam(detection_graph, category_index, is_color_recognition_enabled):

        color = "waiting..."
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

            cap = cv2.VideoCapture(0)
            (ret, frame) = cap.read()

            # for all the frames that are extracted from input video
            while True:
                # Capture frame-by-frame
                (ret, frame) = cap.read()          

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, counting_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if(len(counting_result) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    cv2.putText(input_frame, counting_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

def targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_object):

        # input video
        cap = cv2.VideoCapture(input_video)

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        total_passed_objects = 0
        color = "waiting..."
        the_result = "..."
        height = 0
        width = 0
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

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      targeted_objects=targeted_object,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if(len(the_result) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    cv2.putText(input_frame, the_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                #cv2.imshow('object counting',input_frame)

                output_movie.write(input_frame)
                print ("writing frame")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

def single_image_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled):     
        color = "waiting..."
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

        input_frame = cv2.imread(input_video)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(input_frame, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # insert information text to video frame
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Visualization of the results of a detection.        
        counter, csv_line, counting_result = vis_util.visualize_boxes_and_labels_on_single_image_array(1,input_frame,
                                                                                              is_color_recognition_enabled,
                                                                                              np.squeeze(boxes),
                                                                                              np.squeeze(classes).astype(np.int32),
                                                                                              np.squeeze(scores),
                                                                                              category_index,
                                                                                              use_normalized_coordinates=True,
                                                                                              line_thickness=4)
        if(len(counting_result) == 0):
            cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
        else:
            cv2.putText(input_frame, counting_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
        
        cv2.imshow('tensorflow_object counting_api',input_frame)        
        cv2.waitKey(0)

        return counting_result

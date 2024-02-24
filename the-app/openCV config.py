import cv2
import numpy as np
import mediapipe as mp

#cv2 open test for lunge
cap = cv2.VideoCapture("./videos/Lunges/Lunge1.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


#config packet for CalculatorGraph
config_text = {
    input_stream : 'in_stream'
    output_stream : 'out_stream'
    node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in_stream'
        output_stream: 'out_stream'
    }
}




graph = mp.CalculatorGraph(graph_config = config_text)
output_packets = []
graph.observe_output_stream(
    'out_stream',
    lambda stream_name, packet:
        output_packets.append(mp.packet_getter.get_str(packet)))

graph.start_run()

graph.add_packet_to_input_stream('in_stream',
                                 mp.packet_creator.create_string('abc').at(0))

rgb_img = cv2.cvtColor(cv2.imread("./videos/Lunges/Lunge1.mp4"), cv2.COLOR_BGR2RGB,
                       graph.add_packet_to_input_stream(
                           'in_stream',
                           mp.packet_creator.create_image_frame(image_format=mp.ImageFormat.SRGB,
                                                                data=rgb_img).at(1)
                       ))
graph.close()
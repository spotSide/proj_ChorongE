# depth.py

import cv2
import numpy as np
import openvino as ov
import openvino.properties as props
from pathlib import Path
import time
import matplotlib.cm
from numpy.lib.stride_tricks import as_strided
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
utils_dir = os.path.join(parent_dir, "utils")
sys.path.append(utils_dir)
import notebook_utils as utils


class DepthProcessor:
    def __init__(self, compiled_model, input_key, output_key):
        self.compiled_model = compiled_model
        self.input_key = input_key
        self.output_key = output_key

    def process_frame(self, frame):
        """주어진 프레임에서 뎁스 결과를 생성합니다."""
        resized_frame = cv2.resize(frame, (self.input_key.shape[2], self.input_key.shape[3]))
        input_image = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), 0)
        result = self.compiled_model([input_image])[self.output_key]
        return result

    def visualize_result(self, result):
        """뎁스 결과를 시각화합니다."""
        result_frame = self.convert_result_to_image(result)
        return result_frame

    @staticmethod
    def normalize_minmax(data):
        """뎁스 데이터를 정규화합니다."""
        return (data - data.min()) / (data.max() - data.min())

    def convert_result_to_image(self, result, colormap="viridis"):
        """뎁스 결과를 컬러맵으로 변환합니다."""
        cmap = matplotlib.colormaps[colormap]
        result = result.squeeze(0)
        result = self.normalize_minmax(result)
        result = cmap(result)[:, :, :3] * 255
        result = result.astype(np.uint8)
        return result


def download_midas_model():
    """Download and set up the MiDaS Depth Estimation model."""
    model_folder = Path("model/midas")
    model_folder.mkdir(parents=True, exist_ok=True)

    ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
    ir_model_name_xml = "MiDaS_small.xml"
    ir_model_name_bin = "MiDaS_small.bin"

    # Download model files if not already present
    if not (model_folder / ir_model_name_xml).exists():
        utils.download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=model_folder)
    if not (model_folder / ir_model_name_bin).exists():
        utils.download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=model_folder)

    model_xml_path = model_folder / ir_model_name_xml
    print(f"MiDaS model downloaded at: {model_xml_path}")
    return model_xml_path


def process_depth_sections(depth_map, num_rows=4, num_cols=4, threshold=0.8):
    h, w = depth_map.shape
    section_height = h // num_rows
    section_width = w // num_cols
    
    left_count = 0
    right_count = 0

    for row in range(num_rows):
        for col in range(num_cols):
            y1, y2 = row * section_height, (row + 1) * section_height
            x1, x2 = col * section_width, (col + 1) * section_width
            section = depth_map[y1:y2, x1:x2]
            mean_depth = section.mean()
            if mean_depth >= threshold:
                if col < num_cols // 2:
                    left_count += 1
                else:
                    right_count += 1

    if left_count > right_count:
        return "Avoid to Right"
    elif right_count > left_count:
        return "Avoid to Left"
    else:
        return "Balanced"


def display_depth_sections(image, depth_map, num_rows=4, num_cols=4, output_width=1280, output_height=720):
    image = cv2.resize(image, (output_width, output_height))
    depth_map = cv2.resize(depth_map, (output_width, output_height))

    section_height = output_height // num_rows
    section_width = output_width // num_cols

    for row in range(num_rows):
        for col in range(num_cols):
            y1, y2 = row * section_height, (row + 1) * section_height
            x1, x2 = col * section_width, (col + 1) * section_width
            section = depth_map[y1:y2, x1:x2]
            mean_depth = section.mean()

            cv2.putText(
                image,
                f"{mean_depth:.2f}",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return image


def setup_depth_model():
    core = ov.Core()
    model_path = download_midas_model()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model=model, device_name="GPU")
    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    
    return DepthProcessor(compiled_model, input_key, output_key)


def unified_depth(webcam_processor):
    depth_processor = setup_depth_model()

    try:
        while True:
            frame = webcam_processor.read_frame()

            depth_result = depth_processor.process_frame(frame)
            depth_map = (depth_result.squeeze(0) - depth_result.min()) / (depth_result.max() - depth_result.min())
            depth_frame = depth_processor.visualize_result(depth_result)

            decision = process_depth_sections(depth_map, num_rows=4, num_cols=4, threshold=0.8)

            depth_frame_with_sections = display_depth_sections(
                depth_frame.copy(), depth_map, num_rows=4, num_cols=4, output_width=1280, output_height=720
            )

            cv2.putText(
                depth_frame_with_sections,
                decision,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # 빨간색 텍스트
                2,
                cv2.LINE_AA
            )

            depth_frame_with_sections_resized = cv2.resize(depth_frame_with_sections, (1280, 1080))            
            combined_frame = np.hstack((depth_frame_with_sections_resized,))

            cv2.imshow("Depth Estimation", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)

    except Exception as e:
        print(f"Error during unified estimation: {e}")
    finally:
        webcam_processor.release()
        cv2.destroyAllWindows()
import os
from fastapi import FastAPI, HTTPException
from ultralytics import YOLO

app = FastAPI()

# 加载模型（可以在启动时加载一次）
model = YOLO('weights/yolo11n-seg.onnx')  # 使用ONNX模型


@app.post("/crack_segment")
async def crack_segment(filename: str,
                           input_dir: str = "/Users/hongliang/Desktop/yolov12/input",
                           output_dir: str = "/Users/hongliang/Desktop/yolov12/output"):
    # 拼接文件路径
    input_file  = os.path.join(input_dir, filename)
    output_file = os.path.join(output_dir, filename)

    # 检查文件是否存在
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # 进行预测
        results = model.predict(
            source=input_file,
            task='segment',
            imgsz=640,
            save=False,
        )

        for result in results:
            result.save(filename=output_file, boxes=False, conf=False)
            detection = {}
            detection['output_file'] = output_file

            if result.boxes is not None:
                boxes_array = result.boxes.xyxy.cpu().numpy()
                detection['boxes'] = boxes_array.tolist()

            if result.masks is not None:
                detection['masks'] = [mask.tolist() for mask in result.masks.xy]

        return detection


    except ValueError:
        raise HTTPException(status_code=400, detail="Input value error")
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Runtime error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/crack_segment_url")
async def crack_segment_url(url_img: str,
                           output_dir: str = "/Users/hongliang/Desktop/yolov12/output"):
    # 拼接文件路径
    url_name = os.path.basename(url_img)
    output_file = os.path.join(output_dir, url_name)

    try:
        # 进行预测
        results = model.predict(
            source=url_img,
            task='segment',
            imgsz=640,
            save=False,
        )

        for result in results:
            result.save(filename=output_file, boxes=False, conf=False)
            detection = {}
            detection['output_file'] = output_file

            if result.boxes is not None:
                boxes_array = result.boxes.xyxy.cpu().numpy()
                detection['boxes'] = boxes_array.tolist()

            if result.masks is not None:
                detection['masks'] = [mask.tolist() for mask in result.masks.xy]

        return detection


    except ValueError:
        raise HTTPException(status_code=400, detail="Input value error")
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Runtime error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8088)

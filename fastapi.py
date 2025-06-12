from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YOLO墙体裂缝分割API",
    description="接收二进制图片数据，返回分割后的二进制图片",
    version="1.0.0"
)

# 加载YOLO模型
try:
    # 注意：这里的路径需要根据你的实际模型位置调整
    model = YOLO('weights/best.onnx', task='segment')
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    raise RuntimeError(f"模型加载失败: {str(e)}")


@app.post("/predict",
          summary="墙体裂缝分割预测",
          response_description="处理后的二进制图片数据",
          responses={
              200: {"description": "返回处理后的二进制图片,boxes坐标,masks坐标"},
              400: {"description": "无效的输入数据"},
              500: {"description": "服务器处理错误"}
          })
async def predict_image(request: Request):
    """
    接收二进制图片数据，进行分割预测，返回处理后的图片

    参数:
    - request: FastAPI请求对象，包含原始二进制数据

    返回:
    - Response: 包含处理后的二进制图片数据
    """
    try:
        # 1. 读取原始二进制数据
        image_bytes = await request.body()

        # 2. 验证数据是否为空
        if not image_bytes:
            logger.warning("接收到空数据")
            raise HTTPException(status_code=400, detail="图片数据不能为空")

        # 3. 将二进制数据转换为PIL Image对象
        try:
            image = Image.open(io.BytesIO(image_bytes))
            logger.info("图片数据解析成功")
        except Exception as e:
            logger.error(f"图片解析失败: {str(e)}")
            raise HTTPException(status_code=400, detail="无效的图片数据")

        # 4. 转换为OpenCV格式 (BGR)
        try:
            image_np = np.array(image)
            # 处理RGBA图片（如果有alpha通道）
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            else:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            logger.info("图片格式转换成功")
        except Exception as e:
            logger.error(f"图片格式转换失败: {str(e)}")
            raise HTTPException(status_code=400, detail="图片格式转换失败")

        # 5. 使用YOLO模型进行预测
        try:
            results = model.predict(
                source=image_np,
                task='segment',
                imgsz=640,
                save=False,
            )
            logger.info("模型预测完成")
        except Exception as e:
            logger.error(f"模型预测失败: {str(e)}")
            raise HTTPException(status_code=500, detail="模型预测失败")

        # 6. 处理预测结果
        if not results:
            logger.warning("未检测到任何目标")
            raise HTTPException(status_code=400, detail="未检测到任何目标")

        # result = results[0]
        for result in results:
            # result.save(filename=output_file, boxes=False, conf=False)
            detection = {}
            # detection['output_file'] = output_file

            if result.boxes is not None:
                boxes_array = result.boxes.xyxy.cpu().numpy()
                detection['boxes'] = boxes_array.astype(int).tolist()

            if result.masks is not None:
                detection['masks'] = [mask.astype(int).tolist() for mask in result.masks.xy]

        # 7. 绘制预测结果
        try:
            plotted_img = result[0].plot()  # 返回带有预测结果的BGR图像
            logger.info("结果绘制完成")
        except Exception as e:
            logger.error(f"结果绘制失败: {str(e)}")
            raise HTTPException(status_code=500, detail="结果绘制失败")

        # 8. 将结果转换为JPEG二进制
        try:
            _, img_encoded = cv2.imencode('.jpg', plotted_img)
            response_bytes = img_encoded.tobytes()
            detection['img_bytes'] = list(response_bytes)
            logger.info("图片编码完成")
        except Exception as e:
            logger.error(f"图片编码失败: {str(e)}")
            raise HTTPException(status_code=500, detail="图片编码失败")

        # 9. 返回处理后的图片
        # return Response(
        #     content=response_bytes,
        #     media_type="image/jpeg",
        #     headers={"Content-Disposition": "attachment; filename=result.jpg"}
        # )
        return detection

    except HTTPException:
        # 已经处理过的异常，直接抛出
        raise
    except Exception as e:
        logger.error(f"未知错误: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8086, log_level="info")

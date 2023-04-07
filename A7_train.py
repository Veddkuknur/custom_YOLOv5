import numpy as np


def classify_and_detect(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]
    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # add your code here to fill in pred_class and pred_bboxes
    pred_bboxes_list=[]
    pred_class_list=[]
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True).to(device)
    for img_id in range(N):
      src_img = images[img_id, ...].squeeze().reshape((64, 64, 3)).astype(np.uint8)
      results = model(src_img,size=320)
      results = results.xyxy[0].cpu().tolist()
      if len(results)<2:
        temp=[]
        for j in (0,1):
          y_min = results[0][1]
          x_min = results[0][0]
          y_max = results[0][3]
          x_max = results[0][2]
          temp.append([round(y_min),round(x_min),round(y_max),round(x_max)])
        pred_bboxes_list.append(temp)
        pred_class_list.append([results[0][5],results[0][5]])
      elif results[0][5]<results[1][5]:
        temp=[]
        for j in (0,1):
          y_min = results[j][1]
          x_min = results[j][0]
          y_max = results[j][3]
          x_max = results[j][2]
          temp.append([round(y_min),round(x_min),round(y_max),round(x_max)])
        pred_bboxes_list.append(temp)
        pred_class_list.append([results[0][5],results[1][5]])
      else:
        temp=[]
        for j in (1,0):
          y_min = results[j][1]
          x_min = results[j][0]
          y_max = results[j][3]
          x_max = results[j][2]
          temp.append([round(y_min),round(x_min),round(y_max),round(x_max)])
        pred_bboxes_list.append(temp)
        pred_class_list.append([results[1][5],results[0][5]])
        # np.append(pred_bboxes,temp)
        # np.append(pred_class,[results[1][5],results[0][5]])

    pred_class = np.array(pred_class_list, dtype=np.int32)
    pred_bboxes = np.array(pred_bboxes_list, dtype=np.int32)  
    return pred_class, pred_bboxes
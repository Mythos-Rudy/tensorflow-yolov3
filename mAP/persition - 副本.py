# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:16:24 2019

@author: 熊熊熊宇达
"""

import glob
import json
import os
import shutil
import operator
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
parser.add_argument('--width', default=1280, type=int, help="image width.")
parser.add_argument('--height', default=800, type=int, help="image height.")
parser.add_argument('--easing_mode', default=True, type=bool, help="easing_mode or voc_2012 mode.")
parser.add_argument('--max_IOU_thd', default=0.7, type=float, help="IOU compute")
parser.add_argument('--min_IOU_thd', default=0.3, type=float, help="IOU compute")
args = parser.parse_args()

"""
 Plot - adjust axes
"""
def adjust_axes(r, t, fig, axes):
  # get text width for re-scaling
  bb = t.get_window_extent(renderer=r)
  text_width_inches = bb.width / fig.dpi
  # get axis width in inches
  current_fig_width = fig.get_figwidth()
  new_fig_width = current_fig_width + text_width_inches
  propotion = new_fig_width / current_fig_width
  # get axis limit
  x_lim = axes.get_xlim()
  axes.set_xlim([x_lim[0], x_lim[1]*propotion])
  
"""
 Draw plot using Matplotlib
"""
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
  # sort the dictionary by decreasing value, into a list of tuples
  sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
  # unpacking the list of tuples into two lists
  sorted_keys, sorted_values = zip(*sorted_dic_by_value)
  # 
  if true_p_bar != "":
    """
     Special case to draw in (green=true predictions) & (red=false predictions)
    """
    fp_sorted = []
    tp_sorted = []
    for key in sorted_keys:
      fp_sorted.append(dictionary[key] - true_p_bar[key])
      tp_sorted.append(true_p_bar[key])
    plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False')
    plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True', left=fp_sorted)
    # add legend
    plt.legend(loc='lower right')
    """
     Write number on side of bar
    """
    fig = plt.gcf() # gcf - get current figure
    axes = plt.gca()
    r = fig.canvas.get_renderer()
    for i, val in enumerate(sorted_values):
      fp_val = fp_sorted[i]
      tp_val = tp_sorted[i]
      fp_str_val = " " + str(fp_val)
      tp_str_val = fp_str_val + " " + str(tp_val)
      # trick to paint multicolor with offset:
      #   first paint everything and then repaint the first number
      t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
      plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
      if i == (len(sorted_values)-1): # largest bar
        adjust_axes(r, t, fig, axes)
  else:
    plt.barh(range(n_classes), sorted_values, color=plot_color)
    """
     Write number on side of bar
    """
    fig = plt.gcf() # gcf - get current figure
    axes = plt.gca()
    r = fig.canvas.get_renderer()
    for i, val in enumerate(sorted_values):
      str_val = " " + str(val) # add a space before
      if val < 1.0:
        str_val = " {0:.2f}".format(val)
      t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
      # re-set axes to show number inside the figure
      if i == (len(sorted_values)-1): # largest bar
        adjust_axes(r, t, fig, axes)
  # set window title
  fig.canvas.set_window_title(window_title)
  # write classes in y axis
  tick_font_size = 12
  plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
  """
   Re-scale height accordingly
  """
  init_height = fig.get_figheight()
  # comput the matrix height in points and inches
  dpi = fig.dpi
  height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
  height_in = height_pt / dpi
  # compute the required figure height 
  top_margin = 0.15    # in percentage of the figure height
  bottom_margin = 0.05 # in percentage of the figure height
  figure_height = height_in / (1 - top_margin - bottom_margin)
  # set new height
  if figure_height > init_height:
    fig.set_figheight(figure_height)

  # set plot title
  plt.title(plot_title, fontsize=14)
  # set axis titles
  # plt.xlabel('classes')
  plt.xlabel(x_label, fontsize='large')
  # adjust size of window
  fig.tight_layout()
  # save the plot
  fig.savefig(output_path)
  plt.close()
  
def draw_text_in_image(img, text, pos, color, line_width):
  font = cv2.FONT_HERSHEY_PLAIN
  fontScale = 1
  lineType = 1
  bottomLeftCornerOfText = pos
  cv2.putText(img, text,
      bottomLeftCornerOfText,
      font,
      fontScale,
      color,
      lineType)
  text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
  return img, (line_width + text_width)

def error(msg):
  print(msg)
  sys.exit(0)

def file_lines_to_list(path):
  # open txt file lines to a list
  with open(path) as f:
    content = f.readlines()
  # remove whitespace characters like `\n` at the end of each line
  content = [x.strip() for x in content]
  return content

def voc_ap(rec, prec):

    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]

    for i in range(len(mpre)-2, -1, -1):
      mpre[i] = max(mpre[i], mpre[i+1])

    i_list = []
    for i in range(1, len(mrec)):
      if mrec[i] != mrec[i-1]:
        i_list.append(i) 

    ap = 0.0
    for i in i_list:
      ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def boxes_mean(areas):
    mean = np.mean(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)
    box_summary = [mean, min_area, max_area]
    return box_summary

def IOU_threshold(box,box_summary,max_thd=args.max_IOU_thd,min_thd=args.min_IOU_thd,factor=5):
    '''
    根据GT框的大小计算该框的IOU阈值，框越大IOU越接近max_thd，框越小IOU越接近min_thd
    '''
    area = (box[2] - box[0]) * (box[3] - box[1])
    scale = ((area - box_summary[0]) / box_summary[2]) * 2
    sigmoid = 1. / (1 + np.exp(-scale * factor))
    threshold = sigmoid * (max_thd - min_thd) + min_thd
    return threshold

    
def count_gt(save_path,ground_truth_files_list):

    gt_counter_per_class = {}
    areas = []
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt",1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        if not os.path.exists('predicted/' + file_id + ".txt"):
            error_msg = "Error. File not found: predicted/" +  file_id + ".txt\n"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        for line in lines_list:
            try:
                class_name, left, top, right, bottom = line.split()
                areas.append((float(right)-float(left))*(float(bottom)-float(top)))
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error(error_msg)
            
            if class_name in args.ignore:
                continue
            bbox = left + " " + top + " " + right + " " +bottom
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "easy_mode_used":False, "score_used":False, "label_match":False, "IOU":0, "Repeat":0})
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                gt_counter_per_class[class_name] = 1
        with open(save_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)
    gt_classes = sorted(list(gt_counter_per_class.keys()))
    n_classes = len(gt_classes)
    return gt_classes, n_classes, areas, gt_counter_per_class
        
def count_pred(save_path,gt_classes,predicted_files_list):
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in predicted_files_list:
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            if class_index == 0:
                if not os.path.exists('ground-truth/' + file_id + ".txt"):
                     error_msg = "Error. File not found: ground-truth/" +  file_id + ".txt\n"
                     error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
                     error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error(error_msg)
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(save_path + "/" + class_name + "_predictions.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

def comput_ap(class_index, class_name, n_classes, tmp_files_path, box_summary, gt_counter_per_class, easing_mode = True, show_animation = True):
    count_true_positive = 0
    count_iou_positive = 0
    count_wide_positive = 0
    count_neibu_positive = 0
    predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
    predictions_data = json.load(open(predictions_file))
    nd = len(predictions_data)
    tp  = [0] * nd
    dtp = [0] * nd                                                             #匹配GT框被多次检测的
    ktp = [0] * nd                                                             #匹配宽的GT框IOU不够，但宽度够
    ntp = [0] * nd                                                             #匹配预测框再大GT框内部或边缘
    fp  = [0] * nd
    ffp = [0] * nd                                                             #无法满足所有新增positive条件的没救的false框
    tfp = [0] * nd                                                             #解决某些重复框预测为正确导致recall计算偏差问题,easymode匹配过的框算负，tfp+1
    ttp = [0] * nd                                                             #解决某些重复框预测为正确导致recall计算偏差问题，easymode第一次匹配的框才算正，ttp+1
    
    
    for idx, prediction in enumerate(predictions_data):
        file_id = prediction["file_id"]
        img_path = 'images'
        if os.path.exists(img_path): 
          for dirpath, dirnames, files in os.walk(img_path):
            if not files:
                error("Error. Image not found in " + img_path)
        else:
            error("Error. %s not existed "%img_path)
            
        results_files_path = "results"
        if show_animation:
            ground_truth_img = glob.glob1(img_path, file_id + ".*")
            if len(ground_truth_img) == 0:
                error("Error. Image not found with id: " + file_id)
            elif len(ground_truth_img) > 1:
                error("Error. Multiple image with id: " + file_id)
            else: # found image
                # Load image
                img = cv2.imread(img_path + "/" + ground_truth_img[0])
                # load image with draws of multiple detections
                img_cumulative_path = results_files_path + "/images/" + ground_truth_img[0]
                if os.path.isfile(img_cumulative_path):
                    img_cumulative = cv2.imread(img_cumulative_path)
                else:
                    img_cumulative = img.copy()
                bottom_border = 60
                BLACK = [0, 0, 0]
                img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
        
        gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
        ground_truth_data = json.load(open(gt_file))
        ovmax = -1
        gt_match = -1
        match_box = -1
        bb = [ float(x) for x in prediction["bbox"].split() ]
        for obj in ground_truth_data:
            if obj["class_name"] == class_name:
                bbgt = [ float(x) for x in obj["bbox"].split() ]
                bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = obj
                        match_box = bbgt[:]
        
        status = "No box Match!"
        if ovmax < 0:
            ffp[idx] = 1
            fp[idx] = 1
            tfp[idx] = 1
        
        else:
            min_overlap = IOU_threshold(match_box,box_summary)                          #计算IOU阈值（按照框的大小）
    #        if specific_iou_flagged:                                              #可以根据不同label设置不同的IOU阈值
    #            if class_name in specific_iou_classes:
    #                index = specific_iou_classes.index(class_name)
    #                min_overlap = float(iou_list[index])
            if ovmax >= min_overlap:
                #count_iou_positive += 1                                            #只要IOU大于阈值就算iou_TP + 1
                if not bool(gt_match["used"]):                                     #如果GT第一次被匹配
                    if not bool(gt_match["easy_mode_used"]):                   #如果easy条件第一次匹配，则recall计算时为正，否则为负，tfp+1
                        gt_match["easy_mode_used"]  = True
                        ttp[idx] = 1
                    else:
                        tfp[idx] = 1
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positive += 1
                    status = "Matched!"
                else:                                                              #如果不是第一次被匹配但是IOU符合
                    if not bool(gt_match["used"]):                                     #如果GT第一次被匹配
                        if not bool(gt_match["easy_mode_used"]):                   #如果easy条件第一次匹配，则recall计算时为正，否则为负，tfp+1
                            gt_match["easy_mode_used"]  = True
                            ttp[idx] = 1
                        else:
                            tfp[idx] = 1
                    dtp[idx] = 1
                    fp[idx] = 1
                    count_iou_positive += 1
                    status = "Repeat Matched!"
            else:
                fp[idx] = 1
                if ovmax > 0:
                    if match_box[2]-match_box[0] > 0.6 * args.width and (bb[2]-bb[0]) / (match_box[2]-match_box[0]) > 0.6:  #宽框只要宽度够也算ok
                        if not bool(gt_match["easy_mode_used"]):                   #如果easy条件第一次匹配，则recall计算时为正，否则为负，tfp+1
                            gt_match["easy_mode_used"]  = True
                            ttp[idx] = 1
                        else:
                            tfp[idx] = 1                        
                        ktp[idx] = 1
                        count_wide_positive += 1
                        status = "Wide Matched!"
                    elif match_box[3]-match_box[1] > 0.6 * args.height and (bb[3]-bb[1]) / (match_box[3]-match_box[1]) > 0.6:
                        if not bool(gt_match["easy_mode_used"]):                   #如果easy条件第一次匹配，则recall计算时为正，否则为负，tfp+1
                            gt_match["easy_mode_used"]  = True
                            ttp[idx] = 1
                        else:
                            tfp[idx] = 1
                        ktp[idx] = 1
                        count_wide_positive += 1
                        status = "Wide Matched!"
                    elif iw * ih / ((bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)) > min_overlap or iw * ih /((match_box[2]-match_box[0]+1)*(match_box[3]-match_box[1]+1)) > min_overlap:
                        if not bool(gt_match["easy_mode_used"]):                   #如果easy条件第一次匹配，则recall计算时为正，否则为负，tfp+1
                            gt_match["easy_mode_used"]  = True
                            ttp[idx] = 1
                        else:
                            tfp[idx] = 1
                        ntp[idx] = 1
                        count_neibu_positive += 1
                        status = "Inside Matched!"
                    else:
                        ffp[idx] = 1
                        tfp[idx] = 1
                        status = "IOU Not Enough!"
                else:
                    error('逻辑错误')
            with open(gt_file, 'w') as f:
                f.write(json.dumps(ground_truth_data))

        if show_animation:
            height, widht = img.shape[:2]
            white = (255,255,255)
            blue = (255,30,30)
            green = (0,255,0)
            red = (30,30,255)
            pink = (203,192,266)
            Orange= (0,165,255)
            DeepPink = (147,20,255)
            Purple = (204,50,153)
            margin = 10
            v_pos = int(height - margin - (bottom_border / 2))
            text = "Image: " + ground_truth_img[0] + " "
            img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
            text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
            img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), blue, line_width)
            image_color = red
            if ovmax != -1:
                color = green
                if status == "Matched!":
                    image_color = green
                    text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
                elif status == "Repeat Matched!":
                    image_color = pink
                    text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100) + "but repeated matched"
                elif status == "Wide Matched!":
                    image_color = DeepPink
                    text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100) + "but matched wide box"
                elif status == "Inside Matched!":
                    image_color = Purple
                    text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100) + "but pred inside box"
                elif status == "IOU Not Enough!":
                    color,image_color = red,Orange
                    text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100) + "and do not meet others"
                else:
                    error("Error. Unkown status!")
                img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
            # 2nd line
            v_pos += int(bottom_border / 2)
            rank_pos = str(idx+1) # rank position (idx starts at 0)
            text = "Prediction #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(prediction["confidence"])*100)
            img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
            color = red
            if status != "IOU Not Enough!" and status != "No box Match!":
                color = green
            text = "Result: " + status + " "
            img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if ovmax > 0: # if there is intersections between the bounding-boxes
                bbgt = [ int(x) for x in gt_match["bbox"].split() ]
                cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),blue,2)
                cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),blue,2)
                cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, blue, 1, cv2.LINE_AA)
            bb = [int(i) for i in bb]
            cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),image_color,2)
            cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),image_color,2)
            cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, image_color, 1, cv2.LINE_AA)
            # show image
            #cv2.imshow("Animation", img)
            #cv2.waitKey(20) # show for 20 ms
            # save image to results
            output_img_path = results_files_path + "/images/single_predictions/" + class_name + "_prediction" + str(idx) + ".jpg"
            cv2.imwrite(output_img_path, img)
            # save the image with all the objects drawn to it
            cv2.imwrite(img_cumulative_path, img_cumulative)

    
    ftp = [x+y+z+h for x, y, z, h in zip(tp, dtp, ktp, ntp)]
    if len(set(ftp)) > 2:
        print(ftp)
        raise NotImplementedError("exist tp, dtp, ktp, ntp at the same time!")
    if not (all([x+y for x, y in zip(ftp, ffp)]) and any([x+y-1 for x, y in zip(ftp, ffp)])==False):
        print(ftp,'\n',ffp)
        print([x+y for x, y in zip(ftp, ffp)])
        raise NotImplementedError("ftp, fpp errow!")
        
    cumsum = 0
    for idx, val in enumerate(ftp):
        ftp[idx] += cumsum
        cumsum += val
        
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(ffp):
        ffp[idx] += cumsum
        cumsum += val
    
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
    
    cumsum = 0
    for idx, val in enumerate(tfp):
        tfp[idx] += cumsum
        cumsum += val
        
    cumsum = 0
    for idx, val in enumerate(ttp):
        ttp[idx] += cumsum
        cumsum += val
#    print(class_name,'~~~~~~~~~~~~~~~~~')
#    print('tp',tp[-1])
#    print('dtp',dtp[-1])
#    print('ktp',ktp[-1])
#    print('ftp',ftp[-1])
#    print('fp',fp[-1])
#    print('ffp',ffp[-1])
    
    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
    
    frec = ftp[:]
    for idx, val in enumerate(ftp):
        frec[idx] = float(ftp[idx]) / gt_counter_per_class[class_name]
    
    tfrec = ttp[:]
    for idx, val in enumerate(ttp):                                            #解决某些重复框预测为正确导致recall计算偏差问题
        tfrec[idx] = float(ttp[idx]) / gt_counter_per_class[class_name]
    
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        
    fprec = ftp[:]
    for idx, val in enumerate(ftp):
        fprec[idx] = float(ftp[idx]) / (ffp[idx] + ftp[idx])
    

    if easing_mode:
        #fap, fmrec, fmprec = voc_ap(frec, fprec)
        tfap, tfmrec, tfmprec = voc_ap(tfrec, fprec)                           #解决某些重复框预测为正确导致recall计算偏差问题
    else:
        ap, mrec, mprec = voc_ap(rec, prec)

    
    positive_count = [count_true_positive, count_iou_positive, count_wide_positive, count_neibu_positive]
    rounded_prec = [ '%.2f' % elem for elem in prec ]
    rounded_rec = [ '%.2f' % elem for elem in rec ]
    frounded_prec = [ '%.2f' % elem for elem in fprec ]
    frounded_rec = [ '%.2f' % elem for elem in frec ]
    tfrounded_rec = [ '%.2f' % elem for elem in tfrec ]
#    if easing_mode:
#        ap, mrec, mprec, rec, prec, rounded_prec, rounded_rec = fap, fmrec, fmprec, frec, fprec, frounded_prec, frounded_rec

    if easing_mode:                                                             #解决某些重复框预测为正确导致recall计算偏差问题
        ap, mrec, mprec, rec, prec, rounded_prec, rounded_rec = tfap, tfmrec, tfmprec, tfrec, fprec, frounded_prec, tfrounded_rec
        
    return positive_count, ap, mrec, mprec, rec, prec, rounded_prec, rounded_rec
                

def score_prepare(class_name, tmp_files_path, show_animation = True):

    predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
    predictions_data = json.load(open(predictions_file))
    nd = len(predictions_data)

    fp  = [0] * nd                                                           #解决某些重复框预测为正确导致recall计算偏差问题,easymode匹配过的框算负，tfp+1
    
    for idx, prediction in enumerate(predictions_data):
        file_id = prediction["file_id"]
        img_path = 'images'
        if os.path.exists(img_path): 
          for dirpath, dirnames, files in os.walk(img_path):
            if not files:
                error("Error. Image not found in " + img_path)
        else:
            error("Error. %s not existed "%img_path)
            
        results_files_path = "results"
        if show_animation:
            ground_truth_img = glob.glob1(img_path, file_id + ".*")
            if len(ground_truth_img) == 0:
                error("Error. Image not found with id: " + file_id)
            elif len(ground_truth_img) > 1:
                error("Error. Multiple image with id: " + file_id)
            else: # found image
                pass
        
        gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
        ground_truth_data = json.load(open(gt_file))
        ovmax = -1
        gt_match = -1
        match_box = -1
        bb = [ float(x) for x in prediction["bbox"].split() ]
        for obj in ground_truth_data:
            bbgt = [ float(x) for x in obj["bbox"].split() ]
            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1
            if iw > 0 and ih > 0:
                ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih
                ov = iw * ih / ua
                if ov > ovmax:
                    ovmax = ov
                    gt_match = obj
                    match_box = bbgt[:]

        if ovmax < 0:
            fp[idx] = 1
        
        else:
            if not bool(gt_match["score_used"]):                                     #如果GT第一次被匹配
                gt_match["score_used"] = True
                gt_match["IOU"] = ovmax
                if obj["class_name"] == class_name:
                    obj["label_match"] = True

            else:                                                              #如果不是第一次被匹配,Repeat+1
                gt_match["Repeat"] += 1
                if obj["class_name"] == class_name:                            #如果此次分类正确，之前也正确，IOU取更大者，如果之前不正确，则IOU直接取这次的值
                    if obj["label_match"] == True:
                        gt_match["IOU"] = np.maximum(gt_match["IOU"],ovmax)
                    else:
                        gt_match["IOU"] = ovmax
                        obj["label_match"] = True
                else:                                                          #如果此次分类错误，之前也错误，IOU取更大者，如果之前正确，则IOU保持不变
                    if obj["label_match"] != True:
                        gt_match["IOU"] = np.maximum(gt_match["IOU"],ovmax)
            
        with open(gt_file, 'w') as f:
            f.write(json.dumps(ground_truth_data))

            
            
def score_comput(tmp_files_path,box_summary, gt_classes, gt_counter_per_class):
    tmp_files_list = glob.glob(tmp_files_path+'/*_ground_truth.json')
    score_per_class = {}
    matched_num = {}
    unmatched_num = {}
    scores = {}
    for class_name in gt_classes:
        score_per_class[class_name] = 0
        matched_num[class_name] = 0
        unmatched_num[class_name] = 0
        scores[class_name] = 0

    for tmp_file in tmp_files_list:
        data = json.load(open(tmp_file))
        for obj in data:
            IOU = obj["IOU"]
            label_match = obj["label_match"]
            repeat_num = obj["Repeat"]
            label_name = obj["class_name"]
            bbgt = [ float(x) for x in obj["bbox"].split() ]
            min_overlap = IOU_threshold(bbgt,box_summary)                          #计算IOU阈值（按照框的大小）
            label_score = 1 if label_match == True else 0.8
            IOU_score = 1 if IOU >= min_overlap else IOU / min_overlap
            score = IOU_score * label_score
            if label_name in score_per_class:
                score_per_class[label_name] += score
            else:
                score_per_class[label_name] = score
                
            if obj['easy_mode_used'] == True:
                if label_name in matched_num:
                    matched_num[label_name] += 1
                else:
                    matched_num[label_name] = 1
            else:
                if label_name in unmatched_num:
                    unmatched_num[label_name] += 1
                else:
                    unmatched_num[label_name] = 1

    for class_name in gt_classes:
        scores[class_name] = score_per_class[class_name] / gt_counter_per_class[class_name]
    total_score = sum(score_per_class.values()) / sum(gt_counter_per_class.values())
    scores['total'] = total_score
    matched_num['total'] = sum(matched_num.values())
    unmatched_num['total'] = sum(unmatched_num.values())
    
    return scores, matched_num, unmatched_num


def main():
    
    '''
    define visiabliazation
    '''
    if args.ignore is None:
        args.ignore = ['ZK','ZH']
    
#    specific_iou_flagged = False
#    if args.set_class_iou is not None:
#        specific_iou_flagged = True
    
    # if there are no images then no animation can be shown
    
    show_animation = False
    if not args.no_animation:
        show_animation = True
        
    draw_plot = False
    if not args.no_plot:
        draw_plot = True

            
    '''
    define directory
    '''
    tmp_files_path = "tmp_files"
    if not os.path.exists(tmp_files_path):
        os.makedirs(tmp_files_path)
    results_files_path = "results"
    if os.path.exists(results_files_path): # if it exist already
      # reset the results directory
      shutil.rmtree(results_files_path)
    
    os.makedirs(results_files_path)
    if draw_plot:
      os.makedirs(results_files_path + "/classes")
    if show_animation:
      os.makedirs(results_files_path + "/images")
      os.makedirs(results_files_path + "/images/single_predictions")
     
    ground_truth_files_list = glob.glob('ground-truth/*.txt')
    ground_truth_files_list.sort()
      


    gt_classes, n_classes, areas, gt_counter_per_class= count_gt(tmp_files_path,ground_truth_files_list)
    predicted_files_list = glob.glob('predicted/*.txt')
    predicted_files_list.sort()
    count_pred(tmp_files_path,gt_classes,predicted_files_list)
    box_summary = boxes_mean(areas)
    
    
    sum_AP = 0.0
    ap_dictionary = {}
    with open(results_files_path + "/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        count_iou_positives = {}
        count_wide_positives = {}
        count_neibu_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            positive_count, ap, mrec, mprec, rec, prec, rounded_prec, rounded_rec = comput_ap(class_index, class_name, n_classes, tmp_files_path, box_summary, gt_counter_per_class, args.easing_mode)
            sum_AP += ap
            text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
            if not args.quiet:
              print(text)
            ap_dictionary[class_name] = ap
            count_true_positives[class_name] = positive_count[0]
            count_iou_positives[class_name] = positive_count[1]
            count_wide_positives[class_name] = positive_count[2]
            count_neibu_positives[class_name] = positive_count[3]

            
            """
             Draw plot
            """
            if draw_plot:
              plt.plot(rec, prec, '-o')
              # add a new penultimate point to the list (mrec[-2], 0.0)
              # since the last line segment (and respective area) do not affect the AP value
              area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
              area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
              plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
              # set window title
              fig = plt.gcf() # gcf - get current figure
              fig.canvas.set_window_title('AP ' + class_name)
              # set plot title
              plt.title('class: ' + text)
              #plt.suptitle('This is a somewhat long figure title', fontsize=16)
              # set axis titles
              plt.xlabel('Recall')
              plt.ylabel('Precision')
              # optional - set axes
              axes = plt.gca() # gca - get current axes
              axes.set_xlim([0.0,1.0])
              axes.set_ylim([0.0,1.05]) # .05 to give some extra space
              # Alternative option -> wait for button to be pressed
              #while not plt.waitforbuttonpress(): pass # wait for key display
              # Alternative option -> normal display
              #plt.show()
              # save the plot
              fig.savefig(results_files_path + "/classes/" + class_name + ".png")
              plt.cla() # clear axes for next plot
              
              error_ratio = [1-x for x in prec]
              #draw precision, recall，error ratio
              plt.plot(range(len(prec)-2), prec[1:-1],   'go-', label='precision_ratio')
              plt.plot(range(len(rec)-2),  rec[1:-1],    'ro-', label='recall_ratio')
              #plt.plot(range(len(prec)-2), error_ratio[1:-1], color='blue',  label='error_ratio')
              plt.legend(loc='upper right',frameon=False)
              plt.xlabel('prediction No.')
              plt.ylabel('ratio')
              plt.title('Result Analysis')
              fig1 = plt.gcf()
              fig1.savefig(results_files_path + "/classes/" + class_name + "_analysis.png")
              plt.cla() # clear axes for next plot
            
        
        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)

        
    """
     compute scores of each class
    """ 
    for class_index, class_name in enumerate(gt_classes):
        score_prepare(class_name, tmp_files_path)
    scores, matched_numb, unmatched_num = score_comput(tmp_files_path,box_summary,gt_classes, gt_counter_per_class)


    """
     write total number and score of ground-truth per class to results.txt
    """    

    gt_counter_per_class['total'] = sum(gt_counter_per_class.values())
    with open(results_files_path + "/results.txt", 'a') as results_file:
      results_file.write("\n# Number and Score of ground-truth objects per class\n")
      for class_name in sorted(gt_counter_per_class):
        results_file.write(class_name+": "+str(gt_counter_per_class[class_name])+'matched:%d unmatched:%d'%(matched_numb[class_name],
                           unmatched_num[class_name])+'score:'+str(scores[class_name]) + "\n")

    """
     Plot the summary of score of each class
    """
    if draw_plot:
      window_title = "Scores Info"
      plot_title = "Scores of each Class\n"
      x_label = "Scores"
      output_path = results_files_path + "/Score Info.png"
      to_show = False
      plot_color = 'forestgreen'
      draw_plot_func(
        scores,
        len(gt_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
        )
      
    """
     Count number of Predictions
    """    
    pred_counter_per_class = {}
    for txt_file in predicted_files_list:
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            if class_name in args.ignore:
                continue
            if class_name in pred_counter_per_class:
                pred_counter_per_class[class_name] += 1
            else:
                pred_counter_per_class[class_name] = 1
    pred_classes = list(pred_counter_per_class.keys())
    
    """
     Plot the total number of occurences of each class in the ground-truth
    """
    if draw_plot:
      window_title = "Ground-Truth Info"
      plot_title = "Ground-Truth\n"
      plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
      x_label = "Number of objects per class"
      output_path = results_files_path + "/Ground-Truth Info.png"
      to_show = False
      plot_color = 'forestgreen'
      draw_plot_func(
        gt_counter_per_class,
        len(gt_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        matched_numb,
        )

    """
     Write number of predicted objects per class to results.txt
    """
    for class_name in pred_classes:
      # if class exists in predictions but not in ground-truth then there are no true positives in that class
      if class_name not in gt_classes:
        count_true_positives[class_name] = 0
    
    count_easing_true_positives = {}
    for class_name in gt_classes:
        count_easing_true_positives[class_name] = count_true_positives[class_name]+count_iou_positives[class_name]+count_wide_positives[class_name]+count_neibu_positives[class_name]
    
    """
     Plot the total number of occurences of each class in the "predicted" folder
    """
    if draw_plot:
      window_title = "Predicted Objects Info"
      # Plot title
      plot_title = "Predicted Objects\n"
      plot_title += "(" + str(len(predicted_files_list)) + " files and "
      count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter_per_class.values()))
      plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
      # end Plot title
      x_label = "Number of objects per class"
      output_path = results_files_path + "/Predicted Objects Info.png"
      to_show = False
      plot_color = 'forestgreen'
      true_p_bar = count_easing_true_positives
      draw_plot_func(
        pred_counter_per_class,
        len(pred_counter_per_class),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        true_p_bar
        )

    with open(results_files_path + "/results.txt", 'a') as results_file:
      results_file.write("\n# Number of predicted objects per class\n")
      for class_name in sorted(pred_classes):
        n_pred = pred_counter_per_class[class_name]
        text = class_name + ": " + str(n_pred)
        if args.easing_mode:
            text += " (tp:" + str(count_easing_true_positives[class_name]) + ""
            text += " ttp:" + str(count_true_positives[class_name]) + ""
            text += " dtp:" + str(count_iou_positives[class_name]) + ""
            text += " ktp:" + str(count_wide_positives[class_name]) + ""
            text += " ntp:" + str(count_neibu_positives[class_name]) + ""
            text += ", fp:" + str(n_pred-count_true_positives[class_name]-count_iou_positives[class_name]-count_wide_positives[class_name]-count_neibu_positives[class_name]) + ")\n"
        else:
            text +=" (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
        results_file.write(text)
    """
     Draw mAP plot (Show AP's of all classes in decreasing order)
    """
    if draw_plot:
      window_title = "mAP"
      plot_title = "mAP = {0:.2f}%".format(mAP*100)
      x_label = "Average Precision"
      output_path = results_files_path + "/mAP.png"
      to_show = True
      plot_color = 'royalblue'
      draw_plot_func(
        ap_dictionary,
        n_classes,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
        )
        
if __name__ == '__main__':
    main()
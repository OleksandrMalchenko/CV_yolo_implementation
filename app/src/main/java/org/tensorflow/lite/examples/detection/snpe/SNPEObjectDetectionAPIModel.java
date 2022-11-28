
package org.tensorflow.lite.examples.detection.snpe;

import android.app.Application;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;
import android.widget.Toast;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;

import org.tensorflow.lite.examples.detection.deepsort.Deepsort_RBC;
import org.tensorflow.lite.examples.detection.deepsort.Detection;
import org.tensorflow.lite.examples.detection.deepsort.StateInfo;
import org.tensorflow.lite.examples.detection.deepsort.Tracker;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class SNPEObjectDetectionAPIModel implements Classifier {

  private int inputSize;
  public static  int itr = 0;
  private int[] intValues;

  private SNPEObjectDetectionAPIModel() {
  }

  private SNPE.NeuralNetworkBuilder builder;

  private FloatTensor tensor;

  private NeuralNetwork network;

  private NeuralNetwork emb_network;

  private SNPE.NeuralNetworkBuilder builder1;

  private FloatTensor emb_tensor;

  private InputStream is;

  private boolean reset_deepsort = false;

  static int box_count = 0;
  private int ROI_height1;
  private int ROI_height2;
  private int ROI_buffer1;
  private int ROI_buffer2;
  static final String[] inout = {"", "ROI1" , "", "ROI2"};
  private float ROI1;
  private float ROI2;
  Application application;

  private static Deepsort_RBC deepsort_rbc = new Deepsort_RBC();

  private static Map<Integer, StateInfo> my_dict = new HashMap<>();
  private static Map<Integer, Integer> disjoint = new HashMap<>();

  public static Classifier create(
          final AssetManager assetManager,
          final Application application,
          final int inputSize,
          final float ROI1,
          final float ROI2)
          throws IOException {

    final SNPEObjectDetectionAPIModel d = new SNPEObjectDetectionAPIModel();
    d.application = application;
    d.inputSize = inputSize;
    d.ROI1 = ROI1;
    d.ROI2 = ROI2;
    d.ROI_height1 = (int)(d.inputSize * d.ROI1);
    d.ROI_height2 = (int)(d.inputSize * d.ROI2);
    d.ROI_buffer1 = (int)(d.inputSize * 0.10);
    d.ROI_buffer2 = (int)(d.inputSize * 0.05);
    d.intValues = new int[d.inputSize * d.inputSize];

    //InputStream modelInputStream = assetManager.open("best.dlc");
    InputStream modelInputStream = assetManager.open("best_quant.dlc");
    //InputStream modelInputStream = assetManager.open("best_benu.dlc");
    int streamLength = modelInputStream.available();

    d.builder = new SNPE.NeuralNetworkBuilder(application)
            .setCpuFallbackEnabled(true)
            .setRuntimeOrder(NeuralNetwork.Runtime.DSP, NeuralNetwork.Runtime.GPU, NeuralNetwork.Runtime.CPU)
            .setModel(modelInputStream, streamLength)
            .setUseUserSuppliedBuffers(false);

    d.network = d.builder.build();

    d.tensor = d.network.createFloatTensor(1, 320, 320, 3);

    InputStream modelInputStream1 = assetManager.open("nanonets.dlc");
    int streamLength1 = modelInputStream1.available();

    d.builder1 = new SNPE.NeuralNetworkBuilder(application)
            .setCpuFallbackEnabled(true)
            .setRuntimeOrder(NeuralNetwork.Runtime.GPU, NeuralNetwork.Runtime.CPU)
            .setModel(modelInputStream1, streamLength1);

    d.emb_network = d.builder1.build();

    d.emb_tensor = d.emb_network.createFloatTensor(1, 128, 128, 3);

    d.is = assetManager.open("gaus_mask.txt");

    return d;
  }

  private float preProcess(float original) {
    return (original / 255.0f);
  }

  private float[] extractColorChannels(int pixel) {

    float b = ((pixel) & 0xFF);
    float g = ((pixel >> 8) & 0xFF);
    float r = ((pixel >> 16) & 0xFF);

    return new float[]{preProcess(r), preProcess(g), preProcess(b)};
  }

  @Override
  public Pair<List<Recognition>, Integer> recognizeImage(final Bitmap bitmap, long ts) {
//    Bitmap resizedBitmap = getResizedBitmap(bitmap, 320, 320);
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    float[] input = new float[intValues.length * 3];

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
          final int idx = j * bitmap.getWidth() + i;
          final int batchIdx = idx * 3;

          final float[] rgb = extractColorChannels(intValues[idx]);
          input[batchIdx] = rgb[0];
          input[batchIdx + 1] = rgb[1];
          input[batchIdx + 2] = rgb[2];
        }
    }
    String inputTensorName = (String) network.getInputTensorsNames().toArray()[0];

    String[] outputTensorName = new String[1];
    outputTensorName[0] = (String) network.getOutputTensorsNames().toArray()[0];
//    outputTensorName[1] = (String) network.getOutputTensorsNames().toArray()[1];
//    outputTensorName[2] = (String) network.getOutputTensorsNames().toArray()[2];
//    outputTensorName[3] = (String) network.getOutputTensorsNames().toArray()[3];

    tensor.write(input, 0, input.length);

    Map<String, FloatTensor> inputsMap = new HashMap<>();
    inputsMap.put(inputTensorName, tensor);

    final long javaExecuteStart = SystemClock.elapsedRealtime();
    Map<String, FloatTensor> outputsMap = network.execute(inputsMap);
    final long javaExecuteEnd = SystemClock.elapsedRealtime();
    long mJavaExecuteTime = javaExecuteEnd - javaExecuteStart;

    Log.i("robikart detect time", "" + mJavaExecuteTime);

    FloatTensor fboxes = null, fscores = null, fclasses = null;

    for (Map.Entry<String, FloatTensor> output : outputsMap.entrySet()) {
      for (String mOutputLayer : outputTensorName) {
        if (output.getKey().equals(mOutputLayer)) {
          FloatTensor tensor = output.getValue();
          Log.d("model layers", mOutputLayer);
          switch (mOutputLayer) {
            case "output":
              fboxes = tensor;
              break;
            case "Postprocessor/BatchMultiClassNonMaxSuppression_scores":
              fscores = tensor;
              break;
            case "Postprocessor/BatchMultiClassNonMaxSuppression_classes":
              fclasses = tensor;
              break;
//            case "onnx::Reshape_407":
//              fboxes = tensor;
//              break;
//            case "Postprocessor/BatchMultiClassNonMaxSuppression_scores":
//              fscores = tensor;
//              break;
//            case "Postprocessor/BatchMultiClassNonMaxSuppression_classes":
//              fclasses = tensor;
//              break;
          }
        }
      }
    }
////
//    float[] boxes = new float[Objects.requireNonNull(fboxes).getSize()];
//    float[] scores = new float[Objects.requireNonNull(fscores).getSize()];
//    float[] classes = new float[Objects.requireNonNull(fclasses).getSize()];
////
    float[] outputs = new float[Objects.requireNonNull(fboxes).getSize()];
    ///
    //float[] scores = new float[Objects.requireNonNull(fscores).getSize()];
    //float[] classes = new float[Objects.requireNonNull(fclasses).getSize()];
////

    fboxes.read(outputs, 0, outputs.length);
    ///
//    fboxes.read(boxes, 0, boxes.length);
//    fscores.read(scores, 0, scores.length);
//    fclasses.read(classes, 0, classes.length);
    ////

    float imgScaleX = (float)bitmap.getWidth() / PrePostProcessor.mInputWidth;
    float imgScaleY = (float)bitmap.getHeight() / PrePostProcessor.mInputHeight;
//    float ivScaleX = (float)mResultView.getWidth() / bitmap.getWidth();
//    float ivScaleY = (float)mResultView.getHeight() / bitmap.getHeight();
    float ivScaleX = 0.0f;
    float ivScaleY = 0.0f;

    outputs = ReadFromfile("input.txt", this.application.getApplicationContext());
    Log.d("snpe_engine", "1111111: " + outputs[0] + ", " + outputs[1] +
            ", " + outputs[2] + ", " + outputs[3] + ", " + outputs[4] + ", " + outputs[5]);

    final ArrayList<Result> results = PrePostProcessor.outputsToNMSPredictions(outputs, imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);
    Log.d("snpe_engine", "2222222: " + results.size());
    Log.d("snpe_engine", ">>>>>: " + ROI_height1 +", "+ ROI_buffer1 +", " + ROI_height2 +", " + ROI_buffer2);
    final ArrayList<Recognition> recognitions = new ArrayList<>(results.size());

    FloatTensor features = null;

    List<float[]> features_list = new ArrayList<>();
    List<float[]> bboxes = new ArrayList<>();
    List<Float> out_scores = new ArrayList<>();

//    for(int i = 0;i<boxes.length;i = i + 4){
//
//      if (scores[i/4] > 0.3f && classes[i/4]==0.f) {
//        float x1 = boxes[i+1] * inputSize;
//        float y1 = boxes[i] * inputSize;
//        float x2 = boxes[i+3] * inputSize;
//        float y2 = boxes[i+2] * inputSize;
//        Log.d("snpe_engine", ">>>>>>: " + x1+", " + y1+", " + x2+"," + y2);
//
//        if (((( x2 + x1 ) / 2) >( ROI_height1-ROI_buffer1)) && ((( x2 + x1 ) / 2) < (ROI_height2+ROI_buffer2))) {
//          RectF detection = new RectF(x1, y1, x2, y2);
//
//          Bitmap croppedBitmap = getCroppedBitmap(detection);
//
//          croppedBitmap = getResizedBitmap(croppedBitmap);
//
//          float[] emb_input = convertBitmapToFloat(croppedBitmap, is);
//
//          emb_tensor.write(emb_input, 0, emb_input.length);
//
//          String inputTensorName1 = (String) emb_network.getInputTensorsNames().toArray()[0];
//          String[] outputTensorName1 = new String[1];
//          outputTensorName1[0] = (String) emb_network.getOutputTensorsNames().toArray()[0];
//
//          Map<String, FloatTensor> inputsMap1 = new HashMap<>();
//          inputsMap1.put(inputTensorName1, emb_tensor);
//
//          final long javaExecuteStart1 = SystemClock.elapsedRealtime();
//          Map<String, FloatTensor> outputsMap1 = emb_network.execute(inputsMap1);
//          final long javaExecuteEnd1 = SystemClock.elapsedRealtime();
//          mJavaExecuteTime = javaExecuteEnd1 - javaExecuteStart1;
//
//          Log.i("robikart emb net time", "" + mJavaExecuteTime);
//
//          for (Map.Entry<String, FloatTensor> output : outputsMap1.entrySet()) {
//            for (String mOutputLayer : outputTensorName1) {
//              if (output.getKey().equals(mOutputLayer)) {
//                FloatTensor tensor1 = output.getValue();
//                if (mOutputLayer.equals("bn_scale7.batch_norm_blob7"))
//                  features = tensor1;
//              }
//            }
//          }
//
//          float[] feats = features != null ? new float[features.getSize()] : new float[0];
//
//          Objects.requireNonNull(features).read(feats, 0, feats.length);
//
//          features_list.add(feats);
//          float width = (x2) - (x1);
//          float height = (y2) - (y1);
//
//          bboxes.add(new float[]{detection.left, detection.top, width, height});
//          out_scores.add(scores[i / 4]);
////          recognitions.add(
////                  new Recognition(
////                          "",
////                          "",
////                          "",
////                          1.0f,
////                          detection));
//        }
//      }
//    }

    for(int i = 0; i < results.size(); i++) {
      float x1 = results.get(i).rect.left;
      float y1 = results.get(i).rect.top;
      float x2 = results.get(i).rect.right;
      float y2 = results.get(i).rect.bottom;
      Log.d("snpe_engine", "x1, y1, x2, y2: " + x1 + y1 + x2 + y2);

      if (((( x2 + x1 ) / 2) >( ROI_height1-ROI_buffer1)) && ((( x2 + x1 ) / 2) < (ROI_height2+ROI_buffer2))) {
        RectF detection = new RectF(x1, y1, x2, y2);

        Bitmap croppedBitmap = getCroppedBitmap(detection);

        croppedBitmap = getResizedBitmap(croppedBitmap, 128, 128);

        float[] emb_input = convertBitmapToFloat(croppedBitmap, is);

        emb_tensor.write(emb_input, 0, emb_input.length);

        String inputTensorName1 = (String) emb_network.getInputTensorsNames().toArray()[0];
        String[] outputTensorName1 = new String[1];
        outputTensorName1[0] = (String) emb_network.getOutputTensorsNames().toArray()[0];

        Map<String, FloatTensor> inputsMap1 = new HashMap<>();
        inputsMap1.put(inputTensorName1, emb_tensor);

        final long javaExecuteStart1 = SystemClock.elapsedRealtime();
        Map<String, FloatTensor> outputsMap1 = emb_network.execute(inputsMap1);
        final long javaExecuteEnd1 = SystemClock.elapsedRealtime();
        mJavaExecuteTime = javaExecuteEnd1 - javaExecuteStart1;

        Log.i("robikart emb net time", "" + mJavaExecuteTime);

        for (Map.Entry<String, FloatTensor> output : outputsMap1.entrySet()) {
          for (String mOutputLayer : outputTensorName1) {
            if (output.getKey().equals(mOutputLayer)) {
              FloatTensor tensor1 = output.getValue();
              if (mOutputLayer.equals("bn_scale7.batch_norm_blob7"))
                features = tensor1;
            }
          }
        }

        float[] feats = features != null ? new float[features.getSize()] : new float[0];

        Objects.requireNonNull(features).read(feats, 0, feats.length);

        features_list.add(feats);
        float width = (x2) - (x1);
        float height = (y2) - (y1);

        bboxes.add(new float[]{detection.left, detection.top, width, height});
        out_scores.add(results.get(i).score);
//          recognitions.add(
//                  new Recognition(
//                          "",
//                          "",
//                          "",
//                          1.0f,
//                          detection));
      }
    }

    Log.d("snpe_enigne", "444444: " + bboxes.size());
    final long javaExecuteStart1 = SystemClock.elapsedRealtime();
    Pair<Tracker, List<Detection>> track_dets = deepsort_rbc.run_deep_sort(features_list, out_scores, bboxes);
    final long javaExecuteEnd1 = SystemClock.elapsedRealtime();
    mJavaExecuteTime = javaExecuteEnd1 - javaExecuteStart1;

    Log.i("robikart deepsort time", "" + mJavaExecuteTime);

    Object[] result = deepsort_rbc.remap(track_dets.first.mtracks_, my_dict, disjoint);
    List<Integer> track_ids = (List<Integer>) result[0];
    disjoint = (Map<Integer, Integer>) result[1];

    my_dict = deepsort_rbc.delete_removedbox(my_dict, track_ids, disjoint);

    my_dict = deepsort_rbc.update_dict(track_dets.first.mtracks_, my_dict, ts, ROI_height1 , ROI_height2);
    reset_deepsort = false;
    //box occluding the entire screen for scanning
    Log.d("snpe_engine", "Number in dict " + my_dict.size());
    for (Map.Entry<Integer, StateInfo> item : my_dict.entrySet()) {
      if (item.getValue().getUpdate_frame() == ts) {
        float[] bbox = item.getValue().getMbbox();
        RectF tracked = new RectF(bbox[0], bbox[1], bbox[2], bbox[3]);
//        if ((((bbox[0] + bbox[2]) / 2) > ROI_height2) && ((item.getValue().getROI() == 1)||(item.getValue().getROI() == 2)) && item.getValue().isOut_in()) {
//        if ((((bbox[0] + bbox[2]) / 2) > ROI_height2) && (item.getValue().isEntry1()) && (item.getValue().isEntry2())) {
          if ((((bbox[0] + bbox[2]) / 2) > ROI_height2) && (item.getValue().isEntry2())) {
          item.getValue().setROI(3);
          item.getValue().setRandom(false);
          box_count++;
          String tmp = String.format("CV ROI-2 BX N");
          Log.i("Protocol Message", tmp);
          Toast.makeText(application.getApplicationContext(), tmp, Toast.LENGTH_LONG).show();
          reset_deepsort = true;
//        } else if ((((bbox[0] + bbox[1]) / 2) < ROI_height1) && ((item.getValue().getROI() == 2)||(item.getValue().getROI() == 3)) && (item.getValue().isOut_in()==false)) {
        } else if ((((bbox[0] + bbox[1]) / 2) < ROI_height1) && (item.getValue().isEntry1() == false)&& (item.getValue().isEntry2() == false)&& (item.getValue().isExit1())) {
          item.getValue().setROI(1);
          box_count--;
          String tmp = String.format("CV ROI-2 BX E");
          Log.i("Protocol Message", tmp);
          Toast.makeText(application.getApplicationContext(), tmp, Toast.LENGTH_LONG).show();
          reset_deepsort = true;
        } else if ((((bbox[0] + bbox[2]) / 2) < ROI_height1) && (item.getValue().isEntry1()==true)&& (item.getValue().isEntry2()==false)) {
          //if (ts - item.getValue().getFirst_ts() >= 20) {
            String tmp = String.format("CV ROI-1 NEITHER");
            Log.i("Protocol Message", tmp);
            Toast.makeText(application.getApplicationContext(), tmp, Toast.LENGTH_LONG).show();
            reset_deepsort = true;
          //}
        }
        if (reset_deepsort==true){
          deepsort_rbc = new Deepsort_RBC();
          my_dict = new HashMap<>();
          disjoint = new HashMap<>();
        }

        recognitions.add(
                new Recognition(
                        "" + item.getValue().getId(),
                        "T",
                        inout[item.getValue().getROI()],
                        1.0f,
                        tracked
                )
        );
      }
    }

//    for (Map.Entry<Integer, StateInfo> item : my_dict.entrySet()) {
//      if (item.getValue().getUpdate_frame() == ts) {
//        float[] bbox = item.getValue().getMbbox();
//        RectF tracked = new RectF(bbox[0], bbox[1], bbox[2], bbox[3]);
//        recognitions.add(
//                new Recognition(
//                        "" + item.getValue().getId(),
//                        "T",
//                        inout[item.getValue().getROI()],
//                        1.0f,
//                        tracked
//                )
//        );
//      }
//    }
    Log.d("snpe_engine", "55555555: " + recognitions.size() + ", " + box_count);
    return new Pair<List<Recognition>, Integer>(recognitions, box_count);
  }

  private Bitmap getCroppedBitmap(RectF rect) {
    if((rect.right - rect.left)>0 && (rect.bottom - rect.top)>0)
    return Bitmap.createBitmap((int) (rect.right - rect.left), (int) (rect.bottom - rect.top), Bitmap.Config.ARGB_8888);
    else{
      return Bitmap.createBitmap((int)(1), (1), Bitmap.Config.ARGB_8888);
    }
  }

  private float[] convertBitmapToFloat(Bitmap image, InputStream is) {

    float[][] g_mask = new float[128][128];

    try {
      DataInputStream dis = new DataInputStream(is);
      int i, j = 0;
      while (dis.available() > 0) {
        j++;
        i = j % 128;
        if (j == 128) {
          j = 0;
        }
        float a = dis.readFloat();
        g_mask[i][j] = a;
      }
    } catch (Exception e) {
      e.printStackTrace();
    }

    int[] pixValues = new int[image.getWidth() * image.getHeight()];

    image.getPixels(pixValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

    float[] output = new float[pixValues.length * 3];

    for (int i = 0; i < image.getWidth(); ++i) {
      for (int j = 0; j < image.getHeight(); ++j) {
        final int idx = j * image.getWidth() + i;
        final int batchIdx = idx * 3;

        final float[] rgb = extractColorChannels(pixValues[idx]);
        output[batchIdx] = rgb[0] * g_mask[i][j];
        output[batchIdx + 1] = rgb[1] * g_mask[i][j];
        output[batchIdx + 2] = rgb[2] * g_mask[i][j];
      }
    }

    return output;
  }

  private Bitmap getResizedBitmap(Bitmap bm, int w, int h) {
    int width = bm.getWidth();
    int height = bm.getHeight();
    float scaleWidth = ((float) w) / width;
    float scaleHeight = ((float) h) / height;

    Matrix matrix = new Matrix();
    matrix.postScale(scaleWidth, scaleHeight);

    return Bitmap.createScaledBitmap(bm, w, h, false);
  }

  @Override
  public void setNumThreads(int num_threads) {

  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
  }

  public float[] ReadFromfile(String fileName, Context context) {
    float[] preds = new float[30240];
    InputStream fIn = null;
    InputStreamReader isr = null;
    BufferedReader input = null;
    try {
      fIn = context.getResources().getAssets()
              .open(fileName, Context.MODE_WORLD_READABLE);
      isr = new InputStreamReader(fIn);
      input = new BufferedReader(isr);
      String line = "";
      int idx = 0;
      while ((line = input.readLine()) != null) {
        preds[idx] = Float.parseFloat(line);
        idx++;
      }
    } catch (Exception e) {
      e.getMessage();
    } finally {
      try {
        if (isr != null)
          isr.close();
        if (fIn != null)
          fIn.close();
        if (input != null)
          input.close();
      } catch (Exception e2) {
        e2.getMessage();
      }
    }
    return preds;
  }
}
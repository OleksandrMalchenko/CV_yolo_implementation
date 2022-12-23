
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
import java.util.Arrays;
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
  static final String[] classes = {"banana", "banana1", "blackberries", "raspberry", "lemon",
          "lemon1", "grapes", "grapes1", "tomato", "tomato1", "apple", "apple1", "chilli", "chilli1"};
  static final String[] output_classes = {"banana", "banana1", "apple", "apple1"};
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

    d.tensor = d.network.createFloatTensor(1, 640, 640, 3);

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

    tensor.write(input, 0, input.length);

    Map<String, FloatTensor> inputsMap = new HashMap<>();
    inputsMap.put(inputTensorName, tensor);

    final long javaExecuteStart = SystemClock.elapsedRealtime();
    Map<String, FloatTensor> outputsMap = network.execute(inputsMap);
    final long javaExecuteEnd = SystemClock.elapsedRealtime();
    long mJavaExecuteTime = javaExecuteEnd - javaExecuteStart;

    Log.i("robikart detect time", "" + mJavaExecuteTime);

    FloatTensor fboxes = null;

    for (Map.Entry<String, FloatTensor> output : outputsMap.entrySet()) {
      for (String mOutputLayer : outputTensorName) {
        if (output.getKey().equals(mOutputLayer)) {
          FloatTensor tensor = output.getValue();
          Log.d("model layers", mOutputLayer);
          switch (mOutputLayer) {
            case "output":
              fboxes = tensor;
              break;
          }
        }
      }
    }
    float[] outputs = new float[Objects.requireNonNull(fboxes).getSize()];

    fboxes.read(outputs, 0, outputs.length);

    float imgScaleX = (float)bitmap.getWidth() / PrePostProcessor.mInputWidth;
    float imgScaleY = (float)bitmap.getHeight() / PrePostProcessor.mInputHeight;

    float ivScaleX = 0.0f;
    float ivScaleY = 0.0f;

    final ArrayList<Result> results = PrePostProcessor.outputsToNMSPredictions(outputs, imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);
    final ArrayList<Recognition> recognitions = new ArrayList<>(results.size());

    for(int i = 0; i < results.size(); i++) {
      float x1 = results.get(i).rect.left;
      float y1 = results.get(i).rect.top;
      float x2 = results.get(i).rect.right;
      float y2 = results.get(i).rect.bottom;
      int class_id = results.get(i).classIndex;

      if (!Arrays.asList(output_classes).contains(classes[class_id]))
        continue;

      Log.d("snpe_engine", "x1, y1, x2, y2: " + x1 + y1 + x2 + y2);

      if (((( x2 + x1 ) / 2) >( ROI_height1-ROI_buffer1)) && ((( x2 + x1 ) / 2) < (ROI_height2+ROI_buffer2))) {
        RectF detection = new RectF(x1, y1, x2, y2);

        int roi_mode = 0;
        if (((x1 + x2) / 2) > ROI_height2) {
          roi_mode = 3;
        } else if ((((x1 + x2) / 2) < ROI_height1)) {
          roi_mode = 1;
        }
        recognitions.add(
                new Recognition(
                        "",
                        classes[class_id],
                        inout[roi_mode],
                        1.0f,
                        detection));
      }
    }

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
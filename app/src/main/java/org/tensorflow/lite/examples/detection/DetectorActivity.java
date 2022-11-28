package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.media.Image;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Pair;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import androidx.annotation.Nullable;

import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;

import org.tensorflow.lite.examples.detection.customview.AutoFitTextureView;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.snpe.Classifier;
import org.tensorflow.lite.examples.detection.snpe.SNPEObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private static final int TF_OD_API_INPUT_SIZE = 320;
  private static final float ROI1 = 0.0f;
  private static final float ROI2 = 1.0f;
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.35f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size( 1440,1080);
  private static final boolean SAVE_PREVIEW_BITMAP = true;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;
  Bitmap preview1Bitmap = null;  //bitmapForUnitTesting
  int[] preview1ARGB;   //arrayForUnitTesting

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix frameToCropTransformUnitTest;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {

      detector =
          SNPEObjectDetectionAPIModel.create(
              getAssets(),
              getApplication(),
              TF_OD_API_INPUT_SIZE,
              ROI1,
              ROI2
          );

      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    frameToCropTransformUnitTest =
            ImageUtils.getTransformationMatrix(
                    cropSize, cropSize,
                    cropSize, cropSize,
                    sensorOrientation, MAINTAIN_ASPECT);

//    cropToFrameTransform = new Matrix();
//    frameToCropTransform.invert(cropToFrameTransform);
      cropToFrameTransform =
            ImageUtils.getTransformationMatrix(
                    cropSize, cropSize,
                    previewHeight, previewWidth,
            sensorOrientation, MAINTAIN_ASPECT);
    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    int new_height = DESIRED_PREVIEW_SIZE.getHeight();//trackingOverlay.getHeight();
    int new_width = DESIRED_PREVIEW_SIZE.getWidth();//trackingOverlay.getWidth();

    System.out.println(new_height + " " + new_width);
    int ROIheight1 = (int) (((float)new_height) * ROI1);
    trackingOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                final Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(2.0f);
                canvas.drawLine(ROIheight1,0,ROIheight1,new_width, paint);
              }
            });
    int ROIheight2 = (int) (((float)new_height) * ROI2);
    trackingOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                final Paint paint = new Paint();
                paint.setColor(Color.BLUE);
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(2.0f);
                canvas.drawLine(ROIheight2,0,ROIheight2,new_width, paint);
              }
            });
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(new_height, new_width, sensorOrientation);

    // load image for unit testing
    InputStream ims = null;
    try {
      // get input stream
      ims = getAssets().open("preview1.png");
      // load image as Drawable
      preview1Bitmap = BitmapFactory.decodeStream(ims);

      preview1ARGB = new int[preview1Bitmap.getWidth() * preview1Bitmap.getHeight()];
      //preview1Bitmap.getPixels(preview1ARGB, 0, preview1Bitmap.getWidth(), 0, 0, preview1Bitmap.getWidth(), preview1Bitmap.getHeight());
    }
    catch(IOException ex) {
      return;
    }
    finally {
      //Always clear and close
      try {
        if (ims != null) {
          ims.close();
          ims = null;
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  @Override
  protected void processImage() {

    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();


    final Canvas canvas = new Canvas(croppedBitmap);
    //canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    ////
    canvas.drawBitmap(preview1Bitmap, frameToCropTransformUnitTest, null);
    ////
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final Pair<List<Classifier.Recognition>, Integer> results_pair = detector.recognizeImage(croppedBitmap, currTimestamp);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(2.0f);
            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<org.tensorflow.lite.examples.detection.snpe.Classifier.Recognition> mappedRecognitions =
                    new LinkedList<org.tensorflow.lite.examples.detection.snpe.Classifier.Recognition>();

            for (final org.tensorflow.lite.examples.detection.snpe.Classifier.Recognition result : results_pair.first) {
              final RectF location = result.getLocation();
              if (location != null) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);

                mappedRecognitions.add(result);
              }
            }
            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();
            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });


  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  private enum DetectorMode {
    TF_OD_API
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}

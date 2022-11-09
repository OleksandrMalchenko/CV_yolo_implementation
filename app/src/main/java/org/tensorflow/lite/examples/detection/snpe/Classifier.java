/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.snpe;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Pair;

import java.util.List;
import java.util.Map;


/** Generic interface for interacting with different recognition engines. */
public interface Classifier {
  Pair<List<Recognition>, Integer> recognizeImage(Bitmap bitmap, long ts);

  void setNumThreads(int num_threads);

  void setUseNNAPI(boolean isChecked);

  /** An immutable result returned by a Classifier describing what was recognized. */
  class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    private final String state;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public String getState() {
      return state;
    }

    public Recognition(
            final String id, final String title, String state, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.state = state;
      this.location = location;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }
}

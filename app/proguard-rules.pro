# TFLite's interpreter reaches native code through JNI; keep its entry points.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

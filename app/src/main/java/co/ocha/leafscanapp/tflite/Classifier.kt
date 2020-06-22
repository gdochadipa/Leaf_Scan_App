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
package co.ocha.leafscanapp.tflite

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.os.Trace
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.util.*
import java.io.IOException
import java.nio.MappedByteBuffer

abstract class Classifier protected constructor(activity: Activity?, device: Device?, numThreads:Int){

    enum class Model {
        FLOAT_MOBILENET, QUANTIZED_MOBILENET, FLOAT_EFFICIENTNET, QUANTIZED_EFFICIENTNET
    }


    enum class Device{
        CPU, NNAPI, GPU
    }

    private var tfliteModel: MappedByteBuffer? = null

    /**
     * Get the image size along the x axis.
     */
    /**
     * Image size along the x axis.
     */
    val imageSizeX: Int

    /**
     * Get the image size along the y axis.
     */
    /**
     * Image size along the y axis.
     */
    val imageSizeY: Int

    /**
     * Optional GPU delegate for accleration.
     */
    private var gpuDelegate: GpuDelegate? = null

    /**
     * Optional NNAPI delegate for accleration.
     */
    private var nnApiDelegate: NnApiDelegate? = null

    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    protected var tflite: Interpreter?

    /**
     * Options for configuring the Interpreter.
     */
    private val tfliteOptions = Interpreter.Options()

    /**
     * Labels corresponding to the output of the vision model.
     */
    private val labels: List<String>

    /**
     * Input image TensorBuffer.
     */
    private var inputImageBuffer: TensorImage

    /**
     * Output probability TensorBuffer.
     */
    private val outputProbabilityBuffer: TensorBuffer

    /**
     * Processer to apply post processing of the output probability.
     */
    private val probabilityProcessor: TensorProcessor

    class Recognition( val id: String?,val title: String?,val confidence: Float?,private var location: RectF?){
        fun getLocation(): RectF {
            return RectF(location)
        }
        fun setLocation(location: RectF?) {
            this.location = location
        }

        override fun toString(): String {
            var resultString = ""
            if (id != null) {
                resultString += "[$id] "
            }
            if (title != null) {
                resultString += "$title "
            }
            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f)
            }
            if (location != null) {
                resultString += location.toString() + " "
            }
            return resultString.trim { it <= ' ' }
        }
        fun formattedString() = "${title?.capitalize()} - ${String.format("(%.1f%%) ", confidence!! * 100.0f)}"
    }

    fun recognizeImage(bitmap:Bitmap, sensorOrientation:Int ):List<Recognition>{
        Trace.beginSection("recognizeImage")
        Trace.beginSection("loadImage")
        val startTime = SystemClock.uptimeMillis()
        inputImageBuffer = loadImage(bitmap, sensorOrientation)
        val endTime = SystemClock.uptimeMillis()
        Trace.endSection()

        Log.v("Bangkit","Timecost to load the image: " + (endTime - startTime))

        Trace.beginSection("runInference")
        val startForReference = SystemClock.uptimeMillis()
        tflite!!.run(inputImageBuffer.buffer,outputProbabilityBuffer.buffer.rewind())
        val endForReference = SystemClock.uptimeMillis()

        val labeledProbability = TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
            .mapWithFloatValue
        Trace.endSection()

        return getTopProbability(labeledProbability)
    }

    fun close() {
        if (tflite != null) {
            tflite!!.close()
            tflite = null
        }
        if (gpuDelegate != null) {
            gpuDelegate!!.close()
            gpuDelegate = null
        }
        if (nnApiDelegate != null) {
            nnApiDelegate!!.close()
            nnApiDelegate = null
        }
        tfliteModel = null
    }


    private fun loadImage(bitmap: Bitmap, sensorOrientation: Int):TensorImage{
        inputImageBuffer.load(bitmap)

        val cropSize = Math.min(bitmap.width, bitmap.height)
        val numRotation = sensorOrientation/90
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
            .add(Rot90Op(numRotation))
            .add(preprocessNormalizeOp)
            .build()
        return imageProcessor.process(inputImageBuffer)
    }

    protected abstract val modelPath: String?
    protected abstract val labelPath: String?
    protected abstract val preprocessNormalizeOp: TensorOperator?
    protected abstract val postprocessNormalizeOp: TensorOperator?


    companion object{
        private const val  MAX_RESULTS = 3

        @Throws(IOException::class)
        fun create(activity: Activity?, model: Model, device: Device?, numThreads: Int): Classifier {
            return if (model == Model.QUANTIZED_MOBILENET) {
                ClassifierQuantizedMobileNet(activity, device, numThreads)
            } else if (model == Model.QUANTIZED_EFFICIENTNET) {
                ClassifierQuantizedEfficientNet(activity, device, numThreads)
            } else {
                throw UnsupportedOperationException()
            }
        }

        private fun getTopProbability(labelProb: Map<String, Float>):List<Recognition>{
            val pq = PriorityQueue(
                MAX_RESULTS,
                Comparator<Recognition> { lhs, rhs -> // Intentionally reversed to put high confidence at the head of the queue.
                    java.lang.Float.compare(rhs.confidence!!, lhs.confidence!!)
                })
            for ((key, value) in labelProb) {
                pq.add(Recognition("" + key, key, value, null))
            }
            val recognitions = ArrayList<Recognition>()
            val recognitionsSize = Math.min(pq.size, MAX_RESULTS)
            for (i in 0 until recognitionsSize) {
                recognitions.add(pq.poll())
            }
            return recognitions
        }
    }

    init {
        tfliteModel = FileUtil.loadMappedFile(activity!!, modelPath!!)
        when (device) {
            Device.NNAPI -> {
                nnApiDelegate = NnApiDelegate()
                tfliteOptions.addDelegate(nnApiDelegate)
            }
            Device.GPU -> {
                gpuDelegate = GpuDelegate()
                tfliteOptions.addDelegate(gpuDelegate)
            }
            Device.CPU -> {
            }
        }
        tfliteOptions.setNumThreads(numThreads)
        tflite = Interpreter(tfliteModel!!, tfliteOptions)
        labels = FileUtil.loadLabels(activity, labelPath!!)
        val imageTensorIndex = 0
        val imageShape = tflite!!.getInputTensor(imageTensorIndex).shape() // {1, height, width, 3}
        imageSizeY = imageShape[1]
        imageSizeX = imageShape[2]
        val imageDataType = tflite!!.getInputTensor(imageTensorIndex).dataType()
        val probabilityTensorIndex = 0
        val probabilityShape = tflite!!.getOutputTensor(probabilityTensorIndex).shape() // {1, NUM_CLASSES}
        val probabilityDataType = tflite!!.getOutputTensor(probabilityTensorIndex).dataType()

        inputImageBuffer = TensorImage(imageDataType)

        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)

        probabilityProcessor = TensorProcessor.Builder().add(postprocessNormalizeOp).build()
        Log.d("Bangkit","Created a Tensorflow Lite Image Classifier.")
    }



}
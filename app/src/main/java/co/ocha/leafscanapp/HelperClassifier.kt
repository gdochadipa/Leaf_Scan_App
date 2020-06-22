package co.ocha.leafscanapp

import android.app.Activity
import android.graphics.Bitmap
import android.util.Log
import co.ocha.leafscanapp.tflite.Classifier
import com.google.android.gms.common.logging.Logger
import java.util.concurrent.Executors
import java.io.IOException
import java.lang.Error
import java.lang.Exception
import kotlin.system.measureTimeMillis
import kotlin.time.measureTime


data class ClassifierSepc(
    val model:Classifier.Model,
    val device: Classifier.Device,
    val numThreads: Int
)

class HelperClassifier(
    private val activity: Activity,
    private val spec:ClassifierSepc
) {
    private var classifier:Classifier? = null
    private var executors = Executors.newSingleThreadExecutor()

    fun execute(
        bitmap: Bitmap,
        onError:(Exception) -> Unit,
        onResult: (List<Classifier.Recognition>) -> Unit
    ){
        val mainOnError = { e: Exception -> activity.runOnUiThread { onError(e) } }
        val mainOnResult = { r: List<Classifier.Recognition> -> activity.runOnUiThread { onResult(r) } }

        executors.execute{
            createClassfier(mainOnError)
            processImage(bitmap, mainOnResult)
        }
    }

    private fun createClassfier(onError:(Exception) -> Unit){
        if(classifier != null) return

        val (model, device, numThreads) =spec

        if(device === Classifier.Device.GPU && (model === Classifier.Model.QUANTIZED_MOBILENET || model === Classifier.Model.QUANTIZED_EFFICIENTNET)){
            Log.d("Bangkit","Not creating classifier: GPU doesn't support quantized models.")
            onError(IllegalStateException("Error regarding GPU support for Quant models[CHAR_LIMIT=60]"))
            return
        }
        try {
            Log.d("Bangkit","Creating classifier (model=$model, device=$device, numThreads=$numThreads)")
            classifier = Classifier.create(activity, model, device, numThreads)
        }catch (e:Exception){

        }
    }

    private fun processImage(bitmap: Bitmap, onResult: (List<Classifier.Recognition>) -> Unit){
        val currentClass = classifier?:throw IllegalStateException("Classifier is not ready")

        measureTimeMillis {
            val results = currentClass.recognizeImage(bitmap, 0)
            onResult(results)
            Log.d("Bangkit","Result: $results")
        }.also {
            Log.v("Bangkit","Detect: $it ms")
        }
    }
}
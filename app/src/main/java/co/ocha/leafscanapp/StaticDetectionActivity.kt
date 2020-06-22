package co.ocha.leafscanapp

import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import co.ocha.leafscanapp.productsearch.BottomSheetScrimView
import co.ocha.leafscanapp.tflite.Classifier
import com.google.android.material.bottomsheet.BottomSheetBehavior
import java.io.IOException
import java.lang.Exception
import android.app.Activity

class StaticDetectionActivity : AppCompatActivity() {

    private var loading: View? = null
    private var inputImage: ImageView? = null
    private var dotViewContainer:ViewGroup? = null

    private var btmSheetBehavior: BottomSheetBehavior<View>? =null
    private var btmSheetScrimView:BottomSheetScrimView? =null
    private var btmSheetCaption: TextView? = null
    private var btmSheetBest : TextView? = null

    private var inputBitmap:Bitmap ? = null
    private var dotViewSize: Int = 0
    private var selectObjectIndex = 0

    private val classifier by lazy {
        HelperClassifier(this, ClassifierSepc(Classifier.Model.QUANTIZED_EFFICIENTNET,Classifier.Device.CPU,1))
    }

    @SuppressLint("SetTextI18n")
    private fun showSearchResult(result: List<Classifier.Recognition>) {
        loading?.visibility = View.GONE
        if (result.size > 1) {
            val resultString = result
                .subList(1, result.size)
                .foldIndexed("") { index, acc, recognition ->
                    "${acc}${index + 2}. ${recognition.formattedString()}\n"
                }
            btmSheetCaption?.text = resultString
        }


        btmSheetBehavior?.peekHeight = PEEK_HEIGHT
        btmSheetBehavior?.state = BottomSheetBehavior.STATE_EXPANDED
        btmSheetBest?.text = "1. ${result.first().formattedString()}"
    }

    private fun setUpBottomSheet(){
        val btmSheetView = findViewById<View>(R.id.bottom_sheet)
        btmSheetBehavior = BottomSheetBehavior.from(btmSheetView).apply {
            addBottomSheetCallback(
                object :BottomSheetBehavior.BottomSheetCallback(){
                    override fun onSlide(bottomSheet: View, slideOffset: Float) {
                        if (slideOffset.isNaN()){
                            return
                        }
                        val collapsStateHeigh = btmSheetBehavior!!.peekHeight.coerceAtMost(bottomSheet.height)
                        btmSheetScrimView?.updateWithThumbnailTranslate(inputBitmap!!,collapsStateHeigh,slideOffset,bottomSheet)
                    }

                    override fun onStateChanged(bottomSheet: View, newState: Int) {
                        Log.d(TAG,  "Bottom sheet new state: $newState")
                        btmSheetScrimView?.visibility = if (newState==BottomSheetBehavior.STATE_HIDDEN) View.GONE else View.VISIBLE
                    }

                }
            )
            state = BottomSheetBehavior.STATE_HIDDEN
        }
        btmSheetScrimView = findViewById<BottomSheetScrimView>(R.id.bottom_sheet_scrim_view).apply {
            setOnClickListener {
                btmSheetBehavior?.state = BottomSheetBehavior.STATE_HIDDEN
            }
        }
        btmSheetCaption = findViewById(R.id.bottom_sheet_caption)
        btmSheetBest = findViewById(R.id.bottom_sheet_best_match)

    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == Utils.REQUEST_CODE_PHOTO_LIBRARY && resultCode == Activity.RESULT_OK){
            data?.data?.let(::detectObject)
        }else{
            super.onActivityResult(requestCode, resultCode, data)
        }
    }

    override fun onBackPressed() {
        if (btmSheetBehavior?.state != BottomSheetBehavior.STATE_HIDDEN) {
            btmSheetBehavior?.setState(BottomSheetBehavior.STATE_HIDDEN)
        } else {
            super.onBackPressed()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_static_detection)

        loading = findViewById<View>(R.id.loading_view)

        inputImage = findViewById(R.id.input_image_view)
        dotViewContainer = findViewById(R.id.dot_view_container)
        dotViewSize = resources.getDimensionPixelOffset(R.dimen.static_image_dot_view_size)

        setUpBottomSheet()

        findViewById<View>(R.id.close_button).setOnClickListener {
            onBackPressed()
        }

        findViewById<View>(R.id.photo_library_button).setOnClickListener {
            Utils.openImagePicker(this)
        }
        intent?.data?.let(::detectObject)
    }



    private fun detectObject(imageUri:Uri){
        inputImage?.setImageDrawable(null)
        dotViewContainer?.removeAllViews()
        selectObjectIndex = 0

        try {
            inputBitmap = Utils.loadImage(this,imageUri, MAX_IMAGE_D)

        }catch (e: IOException){
            Log.e(TAG, "Failed to load file: $imageUri", e)
            return
        }

        inputImage?.setImageBitmap(inputBitmap)
        loading?.visibility = View.INVISIBLE
        val validBitmap = inputBitmap ?: throw NullPointerException("Bitmap is null!")

        classifier.execute(
            bitmap = validBitmap,
            onError = {
                Toast.makeText(
                    this,
                    "Error regarding GPU support for Quant models[CHAR_LIMIT=60]",
                    Toast.LENGTH_LONG
                ).show()
            },
            onResult = {
                showSearchResult(it)
            }
        )
    }

    companion object{
        private const val  TAG = "BangkitStatic"
        private const val MAX_IMAGE_D = 1024
        private const val PEEK_HEIGHT = 200
    }
}
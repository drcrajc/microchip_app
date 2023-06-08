package com.example.bubbledetector

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.text.SpannableString
import android.text.style.RelativeSizeSpan
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.github.dhaval2404.imagepicker.ImagePicker
import org.checkerframework.checker.units.qual.s
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterFactory
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.MappedByteBuffer


class MainActivity : AppCompatActivity() {
    private var imageView: ImageView? = null
    private var captureAndPredictBtn: Button? = null
    private var resultTextView: TextView? = null
    val ASSOCIATED_AXIS_LABELS = "labels.txt";

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initView()

    }

    private fun initView() {
        resultTextView = findViewById<TextView>(R.id.tvPredicttResult)
        captureAndPredictBtn = findViewById<Button>(R.id.btnPredict)
        imageView = findViewById<ImageView>(R.id.imageView)
        3
        captureAndPredictBtn?.setOnClickListener {
            // Opening popup of camera and gallery menu
            initPicker()

            // Reset the result text view
            resultTextView?.text = ""
        }

    }

    private fun initPicker() {

        // popup are open and after selecting or capturing the image we are giving output in onActivityResult method
        ImagePicker.with(this)
            .crop()                    //Crop image(Optional), Check Customization for more option
            // .compress(1024)			//Final image size will be less than 1 MB(Optional)
            .maxResultSize(
                1080,
                1080
            )    //Final image resolution will be less than 1080 x 1080(Optional)
            .start()
    }


    private fun predictionStart(src: Bitmap) {

        //this lines making work for resize or scale the bitmap relevant to width and height
        val bitmap = Bitmap.createScaledBitmap(src, 224, 224, false)


        // this also method from tensor support library
        var tensorImage = TensorImage.fromBitmap(bitmap)
        val imageProcessor =
            ImageProcessor.Builder().add(NormalizeOp(127.5f, 127.5f))
                .build();
        tensorImage = imageProcessor.process(tensorImage);


        // Now we are defining our requirement of input feature of model  (https://netron.app/)

        /*
        * serving_default_sequential_5_input:0181
          name: serving_default_sequential_5_input:0
           type: uint8[1,224,224,3]
            quantization: 0.007843137718737125 * (q - 127)
            location: 181
            StatefulPartitionedCall:0182
            name: StatefulPartitionedCall:0
            type: uint8[1,2]
            quantization: 0.00390625 * q
            location: 182*/


        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(tensorImage.buffer)


        // this is for output buffering but in starting this are empty
        val probabilityBuffer =
            TensorBuffer.createFixedSize(intArrayOf(1, 2), DataType.FLOAT32);


        // this is processing for model connection
        var tflite: InterpreterApi? = null
        try {
            val tfliteModel: MappedByteBuffer = FileUtil.loadMappedFile(
                this,
                "may2xcp.tflite"
            )
            tflite = InterpreterFactory().create(
                tfliteModel, InterpreterApi.Options()


            )
        } catch (e: IOException) {
            Log.e("tfliteSupport", "Error reading model", e)
        }

        // running our prediction and saving our output in probability buffer
        tflite?.run(tensorImage.getBuffer(), probabilityBuffer.buffer)


        // this code for  getting data from labels.txt
        var associatedAxisLabels = mutableListOf<String>();
        try {
            associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);

        } catch (e: IOException) {
            Log.e("tfliteSupport", "Error reading label file", e);
        }

        // Post-processor which dequantize the result , Normalize our output this is only function
        val probabilityProcessor = TensorProcessor.Builder().build()

        if (null != associatedAxisLabels) {
            // Map of labels and their corresponding probability
            val labels = TensorLabel(
                associatedAxisLabels,
                // here we execute our preprocessor function related to normalization, line related to 131
                probabilityProcessor.process(probabilityBuffer)
            )


            // Create a map to access the result based on label
            val floatMap = labels.mapWithFloatValue

            Log.d("floatmap", "workStart2: " + floatMap)

            val positive = floatMap["1 Positive"]
            val negative = floatMap["0 Negative"]
            //val posWithFour: Double = String.format("%.4f", positive).toDouble()
            val posWithPer: Double = String.format("%.2f", positive?.times(100)).toDouble()
            //val negWithFour: Double = String.format("%.4f", negative).toDouble()
            val negWithPer: Double = String.format("%.2f", negative?.times(100)).toDouble()


            val showingText =
                "Result:\n\n" +
                        "Positive: ${posWithPer}% \n" +
                        "Negative: ${negWithPer}% \n"


            /* val ss1 = SpannableString(showingText)
            ss1.setSpan(RelativeSizeSpan(0.8f), 11, 46, 0) // set size
            ss1.setSpan(RelativeSizeSpan(0.8f), 58, showingText.length, 0) // set size
*/
            resultTextView?.setTextSize(18f)
            resultTextView?.text = showingText


        }


    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            // Image Uri will not be null for RESULT_OK
            val imageUri: Uri = data?.data!!

            // Use Uri object instead of File to avoid storage permissions
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val b =
                    ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, imageUri))
                        .copy(Bitmap.Config.ARGB_8888, true)

                b
            } else {
                MediaStore.Images.Media.getBitmap(contentResolver, imageUri)
            }

            // Set the bitmap in the image view
            imageView?.setImageBitmap(bitmap)

            // Reset the result text view
            resultTextView?.text = ""

            // Perform the image classification
            predictionStart(bitmap)
        } else if (resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Task Cancelled", Toast.LENGTH_SHORT).show()
        }
    }
}
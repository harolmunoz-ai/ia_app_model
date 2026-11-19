package com.example.ia_model

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.ia_model.ui.theme.IA_ModelTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            IA_ModelTheme {
                SkinAssessmentScreen()
            }
        }
    }
}

@Composable
fun SkinAssessmentScreen() {
    val context = LocalContext.current

    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var result by remember { mutableStateOf("No results") }
    var loading by remember { mutableStateOf(false) }

    val interpreter = remember {
        try {
            val fileDescriptor = context.assets.openFd("model.tflite")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            Interpreter(modelBuffer)
        } catch (_: Exception) {
            null
        }
    }

    DisposableEffect(interpreter) {
        onDispose {
            interpreter?.close()
        }
    }

    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            result = "Image selected"
            bitmap = try {
                context.contentResolver.openInputStream(uri)?.use {
                    BitmapFactory.decodeStream(it)
                }
            } catch (_: Exception) {
                null
            }
        }
    }

    Scaffold { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            Text(
                text = "Skin Assessment",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )

            Text(
                text = "TensorFlow Lite Classifier",
                style = MaterialTheme.typography.bodyMedium
            )

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(260.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    if (bitmap != null) {
                        Image(
                            bitmap = bitmap!!.asImageBitmap(),
                            contentDescription = "Selected skin image",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                    } else {
                        Text("Select an image")
                    }
                }
            }

            Button(
                onClick = { galleryLauncher.launch("image/*") },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Select Image")
            }

            Button(
                onClick = {
                    val currentBitmap = bitmap
                    if (currentBitmap == null) {
                        result = "Please select an image first"
                    } else if (interpreter == null) {
                        result = "Failed to load model.tflite"
                    } else {
                        loading = true
                        result = "Classifying..."

                        try {
                            val inputShape = interpreter.getInputTensor(0).shape()
                            val inputHeight = if (inputShape.size > 1 && inputShape[1] > 0) inputShape[1] else 224
                            val inputWidth = if (inputShape.size > 2 && inputShape[2] > 0) inputShape[2] else 224
                            val inputChannels = if (inputShape.size > 3 && inputShape[3] > 0) inputShape[3] else 3

                            val scaledBitmap = Bitmap.createScaledBitmap(currentBitmap, inputWidth, inputHeight, true)
                            val inputBuffer = ByteBuffer.allocateDirect(1 * inputHeight * inputWidth * inputChannels * 4)
                            inputBuffer.order(ByteOrder.nativeOrder())

                            val intValues = IntArray(inputWidth * inputHeight)
                            scaledBitmap.getPixels(intValues, 0, inputWidth, 0, 0, inputWidth, inputHeight)

                            var pixelIndex = 0
                            val totalPixels = inputWidth * inputHeight
                            while (pixelIndex < totalPixels) {
                                val pixel = intValues[pixelIndex++]
                                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                                inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
                            }

                            val outputShape = interpreter.getOutputTensor(0).shape()
                            val numClasses = outputShape.last()

                            val outputBuffer = Array(1) { FloatArray(numClasses) }
                            interpreter.run(inputBuffer, outputBuffer)

                            val predictions = outputBuffer[0]
                            if (numClasses == 1) {
                                val score = predictions[0]
                                val confidence = if (score >= 0.5f) score else 1.0f - score
                                val label = if (score >= 0.5f) "Class 1" else "Class 0"
                                val percent = String.format(Locale.US, "%.2f", confidence * 100f)
                                result = "$label: $percent%"
                            } else {
                                var bestIndex = 0
                                var maxConfidence = predictions[0]
                                for (k in 1 until numClasses) {
                                    if (predictions[k] > maxConfidence) {
                                        maxConfidence = predictions[k]
                                        bestIndex = k
                                    }
                                }
                                val percent = String.format(Locale.US, "%.2f", maxConfidence * 100f)
                                result = "Class $bestIndex: $percent%"
                            }
                        } catch (e: Exception) {
                            result = "Error: ${e.localizedMessage}"
                        } finally {
                            loading = false
                        }
                    }
                },
                enabled = !loading,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Classify Image")
            }

            if (loading) {
                CircularProgressIndicator()
            }

            Text(
                text = "Result",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )

            Text(
                text = result,
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center
            )
        }
    }
}

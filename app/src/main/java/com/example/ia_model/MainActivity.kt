package com.example.ia_model

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.ia_model.ui.theme.IA_ModelTheme
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.custom.CustomImageLabelerOptions
import java.util.Locale

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            IA_ModelTheme {
                MalariaScreen()
            }
        }
    }
}

@Composable
fun MalariaScreen() {

    val context = LocalContext.current

    var selectedUri by remember {
        mutableStateOf<Uri?>(null)
    }

    var bitmap by remember {
        mutableStateOf<Bitmap?>(null)
    }

    var result by remember {
        mutableStateOf("Sin resultados")
    }

    var loading by remember {
        mutableStateOf(false)
    }

    val modelExists = remember {
        context.assets.list("")
            ?.contains("model.tflite") == true
    }

    val labeler = remember(modelExists) {
        if (modelExists) {
            val localModel = LocalModel.Builder()
                .setAssetFilePath("model.tflite")
                .build()

            val options = CustomImageLabelerOptions
                .Builder(localModel)
                .setConfidenceThreshold(0.0f)
                .setMaxResultCount(2)
                .build()

            ImageLabeling.getClient(options)
        } else {
            null
        }
    }

    DisposableEffect(labeler) {
        onDispose {
            labeler?.close()
        }
    }

    val galleryLauncher =
        rememberLauncherForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->

            if (uri != null) {
                selectedUri = uri
                result = "Imagen seleccionada"

                bitmap = try {
                    context.contentResolver
                        .openInputStream(uri)?.use {
                            BitmapFactory.decodeStream(it)
                        }
                } catch (e: Exception) {
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
                text = "Malaria Image Classifier",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )

            Text(
                text = "TensorFlow Lite + ML Kit",
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
                            contentDescription = "Imagen microscopica",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
                        )
                    } else {
                        Text("Selecciona una imagen")
                    }
                }
            }

            Button(
                onClick = {
                    galleryLauncher.launch("image/*")
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Seleccionar imagen")
            }

            Button(
                onClick = {
                    val uri = selectedUri

                    if (uri == null) {
                        result = "Selecciona una imagen primero"
                    } else if (labeler == null) {
                        result = "No se encontro model.tflite"
                    } else {
                        loading = true
                        result = "Clasificando..."

                        try {
                            val image = InputImage.fromFilePath(
                                context,
                                uri
                            )

                            labeler.process(image)
                                .addOnSuccessListener { labels ->

                                    val best = labels.maxByOrNull {
                                        it.confidence
                                    }

                                    result = if (best != null) {
                                        val percent = String.format(
                                            Locale.US,
                                            "%.2f",
                                            best.confidence * 100f
                                        )

                                        "${best.text}: $percent %"
                                    } else {
                                        "Sin predicciones"
                                    }

                                    loading = false
                                }
                                .addOnFailureListener { error ->
                                    result =
                                        "Error: ${error.localizedMessage}"
                                    loading = false
                                }

                        } catch (error: Exception) {
                            result =
                                "Error: ${error.localizedMessage}"
                            loading = false
                        }
                    }
                },
                enabled = !loading,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Clasificar imagen")
            }

            if (loading) {
                CircularProgressIndicator()
            }

            Text(
                text = "Resultado",
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

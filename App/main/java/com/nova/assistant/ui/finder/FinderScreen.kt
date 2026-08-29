package com.nova.assistant.ui.finder

import android.graphics.Bitmap
import android.graphics.RectF
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.ui.res.painterResource
import com.nova.assistant.R
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.compose.foundation.Image

@Composable
fun FinderScreen(
    modifier: Modifier = Modifier,
    viewModel: FinderViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    val liveBitmap by viewModel.liveBitmap.collectAsState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color.Black)
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "NOVA Finder",
            color = Color.White,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        when (val s = state) {
            is FinderState.Idle -> IdleContent(
                frame = liveBitmap,
                onMicTap = viewModel::startListening,
                onTextSearch = viewModel::searchTyped,
            )

            is FinderState.WaitingQuery -> WaitingContent(frame = liveBitmap)

            is FinderState.Searching -> SearchingContent(
                query = s.query,
                frame = s.frame,
            )

            is FinderState.Found -> FoundContent(
                query = s.query,
                frame = s.frame,
                box = s.box,
                direction = s.direction,
                onReset = viewModel::reset,
            )

            is FinderState.NotFound -> NotFoundContent(
                query = s.query,
                onReset = viewModel::reset,
            )

            is FinderState.Error -> ErrorContent(
                message = s.message,
                onReset = viewModel::reset,
            )
        }
    }
}

@Composable
private fun ColumnScope.IdleContent(
    frame: Bitmap?,
    onMicTap: () -> Unit,
    onTextSearch: (String) -> Unit,
) {
    var typedQuery by remember { mutableStateOf("") }

    // Live camera frame
    CameraFrameBox(frame = frame, box = null, modifier = Modifier.weight(1f))

    Spacer(Modifier.height(16.dp))

    // Text input fallback
    OutlinedTextField(
        value = typedQuery,
        onValueChange = { typedQuery = it },
        placeholder = { Text("Type object name…", color = Color.Gray) },
        singleLine = true,
        colors = OutlinedTextFieldDefaults.colors(
            focusedTextColor = Color.White,
            unfocusedTextColor = Color.White,
            focusedBorderColor = Color(0xFF00BCD4),
            unfocusedBorderColor = Color.Gray,
        ),
        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
        keyboardActions = KeyboardActions(onSearch = {
            if (typedQuery.isNotBlank()) { onTextSearch(typedQuery); typedQuery = "" }
        }),
        trailingIcon = {
            IconButton(onClick = {
                if (typedQuery.isNotBlank()) { onTextSearch(typedQuery); typedQuery = "" }
            }) { Icon(Icons.Default.Search, "Search", tint = Color(0xFF00BCD4)) }
        },
        modifier = Modifier
            .fillMaxWidth()
            .semantics { contentDescription = "Type what to find" }
    )

    Spacer(Modifier.height(12.dp))

    // Mic button
    Button(
        onClick = onMicTap,
        shape = CircleShape,
        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00BCD4)),
        modifier = Modifier
            .size(72.dp)
            .semantics { contentDescription = "Say what to find" },
    ) {
        Icon(painterResource(R.drawable.ic_mic), contentDescription = null, tint = Color.White, modifier = Modifier.size(32.dp))
    }

    Spacer(Modifier.height(8.dp))
    Text("Tap mic and say the object name", color = Color.Gray, fontSize = 13.sp)
    Spacer(Modifier.height(8.dp))
}

@Composable
private fun ColumnScope.WaitingContent(frame: Bitmap?) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.4f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(600), RepeatMode.Reverse),
        label = "alpha"
    )

    CameraFrameBox(frame = frame, box = null, modifier = Modifier.weight(1f))
    Spacer(Modifier.height(24.dp))

    Icon(
        painterResource(R.drawable.ic_mic),
        contentDescription = "Listening",
        tint = Color(0xFF00BCD4).copy(alpha = alpha),
        modifier = Modifier.size(64.dp),
    )
    Spacer(Modifier.height(8.dp))
    Text("Listening… say the object name", color = Color.White, fontSize = 16.sp)
    Spacer(Modifier.height(24.dp))
}

@Composable
private fun ColumnScope.SearchingContent(query: String, frame: Bitmap) {
    CameraFrameBox(frame = frame, box = null, modifier = Modifier.weight(1f))
    Spacer(Modifier.height(24.dp))
    CircularProgressIndicator(color = Color(0xFF00BCD4), modifier = Modifier.size(48.dp))
    Spacer(Modifier.height(12.dp))
    Text("Searching for \"$query\"…", color = Color.White, fontSize = 16.sp)
    Spacer(Modifier.height(24.dp))
}

@Composable
private fun ColumnScope.FoundContent(
    query: String, frame: Bitmap, box: RectF, direction: String, onReset: () -> Unit,
) {
    CameraFrameBox(frame = frame, box = box, modifier = Modifier.weight(1f))
    Spacer(Modifier.height(16.dp))
    Text(
        text = "✓  $query found — $direction",
        color = Color(0xFF4CAF50), fontSize = 18.sp, fontWeight = FontWeight.Bold,
        textAlign = TextAlign.Center,
        modifier = Modifier.semantics { contentDescription = "$query found, $direction" }
    )
    Spacer(Modifier.height(16.dp))
    OutlinedButton(
        onClick = onReset,
        border = ButtonDefaults.outlinedButtonBorder.copy(width = 1.dp),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF00BCD4)),
    ) { Text("Try Again") }
    Spacer(Modifier.height(8.dp))
}

@Composable
private fun ColumnScope.NotFoundContent(query: String, onReset: () -> Unit) {
    Spacer(Modifier.weight(1f))
    Text("Could not find \"$query\"", color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.SemiBold, textAlign = TextAlign.Center)
    Spacer(Modifier.height(8.dp))
    Text("Move the camera to cover the area and try again.", color = Color.Gray, fontSize = 14.sp, textAlign = TextAlign.Center)
    Spacer(Modifier.height(24.dp))
    Button(onClick = onReset, colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00BCD4))) {
        Text("Try Again")
    }
    Spacer(Modifier.weight(1f))
}

@Composable
private fun ColumnScope.ErrorContent(message: String, onReset: () -> Unit) {
    Spacer(Modifier.weight(1f))
    Text("Error: $message", color = Color(0xFFFF5252), fontSize = 16.sp, textAlign = TextAlign.Center)
    Spacer(Modifier.height(24.dp))
    Button(onClick = onReset, colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00BCD4))) {
        Text("Try Again")
    }
    Spacer(Modifier.weight(1f))
}

/**
 * Shows [frame] with an optional [box] drawn over it.
 * Box coordinates are normalized [0, 1] and mapped to the composable size.
 */
@Composable
private fun CameraFrameBox(
    frame: Bitmap?,
    box: RectF?,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .background(Color.DarkGray, RoundedCornerShape(8.dp)),
        contentAlignment = Alignment.Center,
    ) {
        if (frame != null) {
            Image(
                bitmap = frame.asImageBitmap(),
                contentDescription = "Camera frame",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            Text("Waiting for camera…", color = Color.Gray, fontSize = 14.sp)
        }

        if (box != null) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val left   = box.left   * size.width
                val top    = box.top    * size.height
                val right  = box.right  * size.width
                val bottom = box.bottom * size.height
                drawRect(
                    color = Color(0xFF4CAF50),
                    topLeft = androidx.compose.ui.geometry.Offset(left, top),
                    size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                    style = Stroke(width = 4.dp.toPx()),
                )
            }
        }
    }
}

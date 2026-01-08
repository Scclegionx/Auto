package com.auto_fe.auto_fe.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.expandVertically
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.layout.ContentScale
import coil.compose.AsyncImage
import android.content.Intent
import android.net.Uri
import android.util.Log
import com.auto_fe.auto_fe.ui.theme.*
import com.auto_fe.auto_fe.service.be.UserGuideService
import com.auto_fe.auto_fe.utils.be.SessionManager
import kotlinx.coroutines.launch

@Composable
fun GuideScreen() {
    val context = LocalContext.current
    val haptic = LocalHapticFeedback.current
    val sessionManager = remember { SessionManager(context) }
    val guideService = remember { UserGuideService() }
    val scope = rememberCoroutineScope()

    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var guides by remember { mutableStateOf<List<UserGuideService.GuideItem>>(emptyList()) }
    var expandedVideo by remember { mutableStateOf<Long?>(null) }

    LaunchedEffect(Unit) {
        isLoading = true
        errorMessage = null

        try {
            val result = guideService.getGuides()

            if (result.isSuccess) {
                guides = result.getOrNull()?.data ?: emptyList()
            } else {
                errorMessage = result.exceptionOrNull()?.message ?: "Không thể tải danh sách hướng dẫn"
            }
        } catch (e: Exception) {
            Log.e("GuideScreen", "Error loading guides", e)
            errorMessage = "Lỗi: ${e.message}"
        } finally {
            isLoading = false
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        AIBackgroundDeep,
                        AIBackgroundSoft
                    )
                )
            )
    ) {
        Canvas(
            modifier = Modifier.fillMaxSize()
        ) {
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        Color.Transparent,
                        Color.Black.copy(alpha = 0.08f)
                    ),
                    center = Offset(size.width / 2f, size.height / 2f),
                    radius = size.minDimension * 1.2f
                ),
                radius = size.minDimension * 1.2f,
                center = Offset(size.width / 2f, size.height / 2f)
            )
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            val headerScale by animateFloatAsState(
                targetValue = 1.0f,
                animationSpec = spring(
                    dampingRatio = Spring.DampingRatioMediumBouncy,
                    stiffness = Spring.StiffnessLow
                ),
                label = "header"
            )

            Card(
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface.copy(alpha = 0.8f)
                ),
                shape = RoundedCornerShape(20.dp),
                elevation = CardDefaults.cardElevation(
                    defaultElevation = 4.dp
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .scale(headerScale)
                    .padding(bottom = 16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "📚 Hướng dẫn sử dụng",
                        style = MaterialTheme.typography.headlineLarge.copy(
                            fontSize = 34.sp,
                            lineHeight = 40.sp,
                            fontWeight = FontWeight.Bold
                        ),
                        color = DarkOnSurface,
                        textAlign = TextAlign.Center
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = "Học cách sử dụng ứng dụng một cách dễ dàng",
                        style = MaterialTheme.typography.bodyLarge.copy(
                            fontSize = 22.sp,
                            lineHeight = 28.sp
                        ),
                        color = DarkOnSurface.copy(alpha = 0.8f),
                        textAlign = TextAlign.Center
                    )
                }
            }

            when {
                isLoading -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator(color = DarkPrimary)
                    }
                }
                errorMessage != null -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.padding(16.dp)
                        ) {
                            Text(
                                text = "⚠️",
                                style = MaterialTheme.typography.headlineLarge,
                                fontSize = 48.sp
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = errorMessage ?: "Có lỗi xảy ra",
                                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 20.sp),
                                color = DarkOnSurface,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
                guides.isEmpty() -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "Chưa có hướng dẫn nào",
                            style = MaterialTheme.typography.bodyLarge.copy(fontSize = 22.sp),
                            color = DarkOnSurface.copy(alpha = 0.7f)
                        )
                    }
                }
                else -> {
                    LazyColumn(
                        modifier = Modifier.fillMaxSize(),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        itemsIndexed(guides) { index, guide ->
                            val isExpanded = expandedVideo == guide.id
                            
                            val cardScale by animateFloatAsState(
                                targetValue = if (isExpanded) 1.02f else 1.0f,
                                animationSpec = spring(
                                    dampingRatio = Spring.DampingRatioMediumBouncy,
                                    stiffness = Spring.StiffnessLow
                                ),
                                label = "card_$index"
                            )
                            
                            val cardAlpha by animateFloatAsState(
                                targetValue = if (isExpanded) 1.0f else 0.9f,
                                animationSpec = tween(300),
                                label = "alpha_$index"
                            )

                            VideoCard(
                                guide = guide,
                                isExpanded = isExpanded,
                                onExpandClick = {
                                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                    expandedVideo = if (isExpanded) null else guide.id
                                },
                                onPlayClick = {
                                    haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                    guide.videoUrl?.let { videoUrl ->
                                        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(videoUrl))
                                        intent.setDataAndType(Uri.parse(videoUrl), "video/*")
                                        try {
                                            context.startActivity(intent)
                                        } catch (e: Exception) {
                                            Log.e("GuideScreen", "Error opening video", e)
                                        }
                                    }
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun VideoCard(
    guide: UserGuideService.GuideItem,
    isExpanded: Boolean,
    onExpandClick: () -> Unit,
    onPlayClick: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.7f)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(
            defaultElevation = if (isExpanded) 8.dp else 2.dp
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(width = 80.dp, height = 60.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .clickable { onPlayClick() }
                ) {
                    if (!guide.thumbnailUrl.isNullOrBlank()) {
                        AsyncImage(
                            model = guide.thumbnailUrl,
                            contentDescription = guide.title,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize()
                        )
                    } else {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Color(0xFF4CAF50)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Filled.PlayArrow,
                                contentDescription = null,
                                tint = Color.White,
                                modifier = Modifier.size(32.dp)
                            )
                        }
                    }
                    
                    // Play icon overlay
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Color.Black.copy(alpha = 0.3f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Filled.PlayArrow,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(32.dp)
                        )
                    }
                }
                
                Spacer(modifier = Modifier.width(12.dp))
                
                Column(
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        text = guide.title ?: "",
                        style = MaterialTheme.typography.titleMedium.copy(
                            fontSize = 22.sp,
                            lineHeight = 28.sp,
                            fontWeight = FontWeight.Bold
                        ),
                        color = DarkOnSurface
                    )
                    
                    if (guide.description != null) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = guide.description,
                            style = MaterialTheme.typography.bodySmall.copy(
                                fontSize = 18.sp,
                                lineHeight = 24.sp
                            ),
                            color = DarkOnSurface.copy(alpha = 0.7f),
                            maxLines = if (isExpanded) Int.MAX_VALUE else 1
                        )
                    }
                }
                
                IconButton(
                    onClick = onExpandClick,
                    modifier = Modifier.size(32.dp)
                ) {
                    Icon(
                        imageVector = if (isExpanded) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                        contentDescription = null,
                        tint = DarkOnSurface.copy(alpha = 0.7f),
                        modifier = Modifier.size(20.dp)
                    )
                }
            }
            
            AnimatedVisibility(
                visible = isExpanded,
                enter = expandVertically() + fadeIn(),
                exit = shrinkVertically() + fadeOut()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 12.dp)
                ) {
                    if (guide.description != null) {
                        Text(
                            text = guide.description,
                            style = MaterialTheme.typography.bodyMedium.copy(
                                fontSize = 20.sp,
                                lineHeight = 26.sp
                            ),
                            color = DarkOnSurface.copy(alpha = 0.8f)
                        )
                        
                        Spacer(modifier = Modifier.height(12.dp))
                    }
                    
                    Button(
                        onClick = onPlayClick,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF4CAF50)
                        ),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(
                            imageVector = Icons.Filled.PlayArrow,
                            contentDescription = null,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Xem video hướng dẫn",
                            fontSize = 22.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }
}


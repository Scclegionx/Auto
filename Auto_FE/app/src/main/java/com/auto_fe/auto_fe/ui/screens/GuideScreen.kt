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
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.geometry.Offset
import com.auto_fe.auto_fe.ui.theme.*

/**
 * Màn hình hướng dẫn sử dụng - Thiết kế đặc biệt cho người già
 * - Font chữ lớn, dễ đọc
 * - Nút bấm to, dễ nhấn
 * - Màu sắc tương phản cao
 * - Hướng dẫn từng bước rõ ràng
 */
@Composable
fun GuideScreen() {
    val haptic = LocalHapticFeedback.current
    var selectedCategory by remember { mutableStateOf(0) }
    var expandedVideo by remember { mutableStateOf<Int?>(null) }
    
    // Danh sách các danh mục hướng dẫn
    val guideCategories = listOf(
        GuideCategory(
            title = "🎤 Cách sử dụng giọng nói",
            icon = Icons.Filled.Settings,
            color = Color(0xFF4CAF50),
            videos = listOf(
                GuideVideo(
                    title = "Cách nói lệnh gửi tin nhắn",
                    description = "Hướng dẫn cách nói để gửi tin nhắn cho người thân",
                    duration = "2 phút",
                    thumbnail = "📱"
                ),
                GuideVideo(
                    title = "Cách nói lệnh gọi điện",
                    description = "Hướng dẫn cách nói để gọi điện cho con cháu",
                    duration = "1.5 phút",
                    thumbnail = "📞"
                ),
                GuideVideo(
                    title = "Cách dừng lệnh khi nhầm",
                    description = "Hướng dẫn cách hủy lệnh khi nói nhầm",
                    duration = "1 phút",
                    thumbnail = "⏹️"
                )
            )
        ),
        GuideCategory(
            title = "📱 Cách sử dụng điện thoại",
            icon = Icons.Filled.Phone,
            color = Color(0xFF2196F3),
            videos = listOf(
                GuideVideo(
                    title = "Cách mở ứng dụng Auto FE",
                    description = "Hướng dẫn tìm và mở ứng dụng trên điện thoại",
                    duration = "3 phút",
                    thumbnail = "🔍"
                ),
                GuideVideo(
                    title = "Cách cấp quyền cho ứng dụng",
                    description = "Hướng dẫn cấp quyền microphone và tin nhắn",
                    duration = "4 phút",
                    thumbnail = "🔐"
                ),
                GuideVideo(
                    title = "Cách sử dụng cửa sổ nổi",
                    description = "Hướng dẫn sử dụng nút tròn nổi trên màn hình",
                    duration = "2.5 phút",
                    thumbnail = "🔘"
                )
            )
        ),
        GuideCategory(
            title = "💊 Quản lý đơn thuốc",
            icon = Icons.Filled.Favorite,
            color = Color(0xFFFF9800),
            videos = listOf(
                GuideVideo(
                    title = "Cách đăng nhập tài khoản",
                    description = "Hướng dẫn đăng nhập để xem đơn thuốc",
                    duration = "3 phút",
                    thumbnail = "👤"
                ),
                GuideVideo(
                    title = "Cách xem đơn thuốc",
                    description = "Hướng dẫn xem danh sách đơn thuốc của mình",
                    duration = "2 phút",
                    thumbnail = "📋"
                ),
                GuideVideo(
                    title = "Cách xem chi tiết thuốc",
                    description = "Hướng dẫn xem thông tin chi tiết từng loại thuốc",
                    duration = "2.5 phút",
                    thumbnail = "💊"
                )
            )
        ),
        GuideCategory(
            title = "❓ Giải đáp thắc mắc",
            icon = Icons.Filled.Info,
            color = Color(0xFF9C27B0),
            videos = listOf(
                GuideVideo(
                    title = "Ứng dụng không nghe được giọng nói",
                    description = "Cách khắc phục khi ứng dụng không nhận diện giọng nói",
                    duration = "3 phút",
                    thumbnail = "🔧"
                ),
                GuideVideo(
                    title = "Không gửi được tin nhắn",
                    description = "Cách khắc phục khi không gửi được tin nhắn",
                    duration = "2.5 phút",
                    thumbnail = "📤"
                ),
                GuideVideo(
                    title = "Liên hệ hỗ trợ",
                    description = "Cách liên hệ với con cháu để được hỗ trợ",
                    duration = "1 phút",
                    thumbnail = "📞"
                )
            )
        )
    )

    // Set up screen background with enhanced gradient
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
        // Enhanced vignetting for depth
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
            // Header với animation
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

            // Category tabs với animation
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                items(guideCategories.size) { index ->
                    val category = guideCategories[index]
                    val isSelected = selectedCategory == index
                    
                    // Animation cho card
                    val cardScale by animateFloatAsState(
                        targetValue = if (isSelected) 1.02f else 1.0f,
                        animationSpec = spring(
                            dampingRatio = Spring.DampingRatioMediumBouncy,
                            stiffness = Spring.StiffnessLow
                        ),
                        label = "card_$index"
                    )
                    
                    val cardAlpha by animateFloatAsState(
                        targetValue = if (isSelected) 1.0f else 0.9f,
                        animationSpec = tween(300),
                        label = "alpha_$index"
                    )

                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = if (isSelected) 
                                category.color.copy(alpha = 0.2f) 
                            else 
                                DarkSurface.copy(alpha = 0.7f)
                        ),
                        shape = RoundedCornerShape(16.dp),
                        elevation = CardDefaults.cardElevation(
                            defaultElevation = if (isSelected) 8.dp else 2.dp
                        ),
                        modifier = Modifier
                            .fillMaxWidth()
                            .scale(cardScale)
                            .alpha(cardAlpha)
                            .clickable {
                                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                selectedCategory = index
                            }
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(20.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            // Icon
                            Icon(
                                imageVector = category.icon,
                                contentDescription = null,
                                tint = category.color,
                                modifier = Modifier.size(32.dp)
                            )
                            
                            Spacer(modifier = Modifier.width(16.dp))
                            
                            // Title
                            Text(
                                text = category.title,
                                style = MaterialTheme.typography.titleLarge.copy(
                                    fontSize = 26.sp,
                                    lineHeight = 32.sp,
                                    fontWeight = FontWeight.Bold
                                ),
                                color = DarkOnSurface,
                                modifier = Modifier.weight(1f)
                            )
                            
                            // Arrow
                            Icon(
                                imageVector = if (isSelected) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                                contentDescription = null,
                                tint = DarkOnSurface.copy(alpha = 0.7f),
                                modifier = Modifier.size(24.dp)
                            )
                        }
                    }

                    // Video list với animation
                    AnimatedVisibility(
                        visible = isSelected,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(start = 16.dp, end = 16.dp, bottom = 8.dp)
                        ) {
                            category.videos.forEachIndexed { videoIndex, video ->
                                VideoCard(
                                    video = video,
                                    isExpanded = expandedVideo == videoIndex,
                                    onExpandClick = {
                                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                        expandedVideo = if (expandedVideo == videoIndex) null else videoIndex
                                    },
                                    onPlayClick = {
                                        haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                                        // TODO: Implement video playback
                                    }
                                )
                                
                                if (videoIndex < category.videos.size - 1) {
                                    Spacer(modifier = Modifier.height(8.dp))
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun VideoCard(
    video: GuideVideo,
    isExpanded: Boolean,
    onExpandClick: () -> Unit,
    onPlayClick: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.6f)
        ),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(
            defaultElevation = 1.dp
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            // Video header
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Play button
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFF4CAF50)
                    ),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier
                        .size(48.dp)
                        .clickable { onPlayClick() }
                ) {
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier.fillMaxSize()
                    ) {
                        Text(
                            text = video.thumbnail,
                            fontSize = 24.sp
                        )
                    }
                }
                
                Spacer(modifier = Modifier.width(12.dp))
                
                // Video info
                Column(
                    modifier = Modifier.weight(1f)
                ) {
                        Text(
                            text = video.title,
                            style = MaterialTheme.typography.titleMedium.copy(
                                fontSize = 22.sp,
                                lineHeight = 28.sp,
                                fontWeight = FontWeight.Bold
                            ),
                        color = DarkOnSurface
                    )
                    
                    Spacer(modifier = Modifier.height(4.dp))
                    
                        Text(
                            text = video.duration,
                            style = MaterialTheme.typography.bodySmall.copy(
                                fontSize = 18.sp,
                                lineHeight = 24.sp
                            ),
                        color = DarkOnSurface.copy(alpha = 0.7f)
                    )
                }
                
                // Expand button
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
            
            // Video description (expanded)
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
                        Text(
                            text = video.description,
                            style = MaterialTheme.typography.bodyMedium.copy(
                                fontSize = 20.sp,
                                lineHeight = 26.sp
                            ),
                        color = DarkOnSurface.copy(alpha = 0.8f),
                            lineHeight = 26.sp
                    )
                    
                    Spacer(modifier = Modifier.height(12.dp))
                    
                    // Play button
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

// Data classes
data class GuideCategory(
    val title: String,
    val icon: ImageVector,
    val color: Color,
    val videos: List<GuideVideo>
)

data class GuideVideo(
    val title: String,
    val description: String,
    val duration: String,
    val thumbnail: String
)

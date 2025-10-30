package com.auto_fe.auto_fe.ui.components

import androidx.compose.animation.core.*
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.Spring
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.draw.alpha
import com.auto_fe.auto_fe.ui.theme.*

/**
 * Bottom Navigation với 3 nút: Đơn thuốc/Auth, Ghi âm (chính), Hướng dẫn
 * Nút ghi âm ở giữa có style đặc biệt và nổi bật hơn
 */
@Composable
fun CustomBottomNavigation(
    selectedTab: Int,
    onTabSelected: (Int) -> Unit,
    isLoggedIn: Boolean = false
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.9f)
        ),
        shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Nút Đơn thuốc / Đăng nhập (trái)
            BottomNavItem(
                icon = if (isLoggedIn) "💊" else "🔐",
                label = if (isLoggedIn) "Đơn thuốc" else "Đăng nhập",
                isSelected = selectedTab == 0,
                onClick = { onTabSelected(0) },
                modifier = Modifier.weight(1f)
            )
            
            // Nút Ghi âm (giữa trái) - Style đặc biệt
            BottomNavItemSpecial(
                icon = "🎤",
                label = "Ghi âm",
                isSelected = selectedTab == 1,
                onClick = { onTabSelected(1) },
                modifier = Modifier.weight(1.1f)
            )
            
            // Nút Hướng dẫn (giữa phải)
            BottomNavItem(
                icon = "📚",
                label = "Hướng dẫn",
                isSelected = selectedTab == 2,
                onClick = { onTabSelected(2) },
                modifier = Modifier.weight(1f)
            )
            
            // Nút Cài đặt (phải)
           BottomNavItem(
               icon = "⚙️",
               label = "Cài đặt",
               isSelected = selectedTab == 3,
               onClick = { onTabSelected(3) },
               modifier = Modifier.weight(1f)
           )
        }
    }
}

@Composable
fun BottomNavItem(
    icon: String,
    label: String,
    isSelected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val scale by animateFloatAsState(
        targetValue = if (isSelected) 1.1f else 1.0f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "scale"
    )
    
    val alpha by animateFloatAsState(
        targetValue = if (isSelected) 1.0f else 0.7f,
        animationSpec = tween(200),
        label = "alpha"
    )

    Column(
        modifier = modifier
            .clickable { onClick() }
            .scale(scale)
            .alpha(alpha)
            .padding(vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = icon,
            fontSize = 24.sp,
            modifier = Modifier.padding(bottom = 4.dp)
        )
        
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = if (isSelected) DarkPrimary else DarkOnSurface.copy(alpha = 0.7f),
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
        )
    }
}

@Composable
fun BottomNavItemSpecial(
    icon: String,
    label: String,
    isSelected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val scale by animateFloatAsState(
        targetValue = if (isSelected) 1.2f else 1.0f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "scale"
    )
    
    val pulseScale by animateFloatAsState(
        targetValue = if (isSelected) 1.1f else 1.0f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioLowBouncy,
            stiffness = Spring.StiffnessVeryLow
        ),
        label = "pulse"
    )

    Column(
        modifier = modifier
            .clickable { onClick() }
            .scale(scale)
            .padding(vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Background circle với gradient đặc biệt
        Box(
            modifier = Modifier
                .size(64.dp)
                .scale(pulseScale)
                .clip(CircleShape)
                .background(
                    brush = if (isSelected) {
                        Brush.radialGradient(
                            colors = listOf(
                                DarkPrimary,
                                DarkPrimary.copy(alpha = 0.8f)
                            )
                        )
                    } else {
                        Brush.radialGradient(
                            colors = listOf(
                                DarkSurface.copy(alpha = 0.5f),
                                DarkSurface.copy(alpha = 0.3f)
                            )
                        )
                    }
                ),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = icon,
                fontSize = 28.sp,
                color = if (isSelected) Color.White else DarkOnSurface.copy(alpha = 0.8f)
            )
        }
        
        Spacer(modifier = Modifier.height(4.dp))
        
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = if (isSelected) DarkPrimary else DarkOnSurface.copy(alpha = 0.7f),
            fontWeight = FontWeight.Bold
        )
    }
}
package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.auto_fe.auto_fe.ui.theme.*

@Composable
fun MedicationTabScreen(
    accessToken: String,
    currentUserId: Long? = null,  // Add current user ID
    onPrescriptionClick: (Long) -> Unit,
    onCreatePrescriptionClick: () -> Unit = {},
    onCreateStandaloneMedicationClick: () -> Unit = {},
    onChatClick: () -> Unit = {},
    onLogout: () -> Unit = {},
    onProfileClick: () -> Unit = {},
    onNotificationHistoryClick: () -> Unit = {},
    onEmergencyContactClick: () -> Unit = {},
    onManageConnectionsClick: () -> Unit = {},
    userName: String = "User",
    userEmail: String = "",
    userAvatar: String? = null,
    elderUserId: Long? = null,  // Nếu có = Supervisor đang xem Elder
    elderUserName: String? = null,  // Tên Elder
    onBackClick: (() -> Unit)? = null  // Back về danh sách Elder
) {
    var selectedTabIndex by remember { mutableStateOf(0) }
    val tabs = listOf("📋 Đơn thuốc", "💊 Thuốc ngoài đơn", "📊 Lịch sử uống thuốc")
    val isSupervisorMode = elderUserId != null  // Supervisor mode detection

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
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            // Header Card với Avatar và Dropdown Menu
            HeaderCard(
                userName = userName,
                userEmail = userEmail,
                userAvatar = userAvatar,
                isSupervisorMode = isSupervisorMode,  // Pass supervisor mode
                onLogout = onLogout,
                onProfileClick = onProfileClick,
                onNotificationHistoryClick = onNotificationHistoryClick,
                onEmergencyContactClick = onEmergencyContactClick,
                onManageConnectionsClick = onManageConnectionsClick
            )

            // Tab Row
            TabRow(
                selectedTabIndex = selectedTabIndex,
                containerColor = DarkSurface.copy(alpha = 0.6f),
                contentColor = DarkPrimary,
                modifier = Modifier.padding(horizontal = 16.dp)
            ) {
                tabs.forEachIndexed { index, title ->
                    Tab(
                        selected = selectedTabIndex == index,
                        onClick = { selectedTabIndex = index },
                        text = {
                            Text(
                                text = title,
                                fontWeight = if (selectedTabIndex == index) FontWeight.Bold else FontWeight.Normal,
                                color = if (selectedTabIndex == index) DarkPrimary else DarkOnSurface.copy(alpha = 0.6f)
                            )
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Content based on selected tab
            when (selectedTabIndex) {
                0 -> PrescriptionListTab(
                    accessToken = accessToken,
                    elderUserId = elderUserId,  // Pass elderUserId
                    elderUserName = elderUserName,  // Pass elderUserName
                    onPrescriptionClick = onPrescriptionClick,
                    onCreateClick = onCreatePrescriptionClick,
                    onChatClick = if (isSupervisorMode) ({}) else onChatClick  // Disable chat in supervisor mode
                )
                1 -> StandaloneMedicationTab(
                    accessToken = accessToken,
                    elderUserId = elderUserId,  // Pass elderUserId
                    elderUserName = elderUserName,  // Pass elderUserName
                    onCreateClick = onCreateStandaloneMedicationClick
                )
                2 -> MedicationLogTab(
                    accessToken = accessToken,
                    currentUserId = currentUserId,  // Pass current user ID
                    elderUserId = elderUserId,
                    elderUserName = elderUserName
                )
            }
        }
    }
}

@Composable
private fun HeaderCard(
    userName: String,
    userEmail: String,
    userAvatar: String?,
    isSupervisorMode: Boolean,  // Add supervisor mode flag
    onLogout: () -> Unit,
    onProfileClick: () -> Unit,
    onNotificationHistoryClick: () -> Unit,
    onEmergencyContactClick: () -> Unit,
    onManageConnectionsClick: () -> Unit
) {
    var showMenu by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.9f)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Quản lý thuốc",
                    fontSize = AppTextSize.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Xin chào, ${userName.takeIf { it.isNotBlank() } ?: "Người dùng"}",
                    fontSize = AppTextSize.bodyMedium,
                    color = DarkOnSurface.copy(alpha = 0.7f)
                )
            }

            // Avatar với Dropdown Menu (ẩn khi Supervisor mode)
            if (!isSupervisorMode) {  // Only show menu for Elder
                UserMenu(
                    userAvatar = userAvatar,
                    userName = userName,
                    userEmail = userEmail,
                    showMenu = showMenu,
                    onShowMenuChange = { showMenu = it },
                    onProfileClick = onProfileClick,
                    onNotificationHistoryClick = onNotificationHistoryClick,
                    onEmergencyContactClick = onEmergencyContactClick,
                    onManageConnectionsClick = onManageConnectionsClick,
                    onLogout = onLogout
                )
            }
        }
    }
}

@Composable
private fun UserMenu(
    userAvatar: String?,
    userName: String,
    userEmail: String,
    showMenu: Boolean,
    onShowMenuChange: (Boolean) -> Unit,
    onProfileClick: () -> Unit,
    onNotificationHistoryClick: () -> Unit,
    onEmergencyContactClick: () -> Unit,
    onManageConnectionsClick: () -> Unit,
    onLogout: () -> Unit
) {
    Box {
        Box(
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(DarkPrimary.copy(alpha = 0.2f))
                .clickable { onShowMenuChange(!showMenu) },
            contentAlignment = Alignment.Center
        ) {
            if (!userAvatar.isNullOrBlank()) {
                AsyncImage(
                    model = userAvatar,
                    contentDescription = "User Avatar",
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(CircleShape)
                )
            } else {
                Icon(
                    imageVector = Icons.Default.Person,
                    contentDescription = "User Menu",
                    tint = DarkPrimary,
                    modifier = Modifier.size(28.dp)
                )
            }
        }

        DropdownMenu(
            expanded = showMenu,
            onDismissRequest = { onShowMenuChange(false) },
            modifier = Modifier
                .background(DarkSurface)
                .width(280.dp)
        ) {
            // User Info
            Column(
                modifier = Modifier
                    .padding(16.dp, 12.dp)
                    .fillMaxWidth()
            ) {
                Text(
                    text = userName.takeIf { it.isNotBlank() } ?: "Người dùng",
                    fontSize = AppTextSize.bodyMedium,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                if (userEmail.isNotEmpty()) {
                    Text(
                        text = userEmail,
                        fontSize = AppTextSize.bodySmall,
                        color = DarkOnSurface.copy(alpha = 0.6f),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }

            Divider(color = DarkOnSurface.copy(alpha = 0.2f))

            DropdownMenuItem(
                text = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Person,
                            contentDescription = null,
                            tint = DarkOnSurface,
                            modifier = Modifier.size(20.dp)
                        )
                        Text("Hồ sơ", color = DarkOnSurface)
                    }
                },
                onClick = {
                    onShowMenuChange(false)
                    onProfileClick()
                }
            )

            DropdownMenuItem(
                text = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Notifications,
                            contentDescription = null,
                            tint = DarkOnSurface,
                            modifier = Modifier.size(20.dp)
                        )
                        Text("Lịch sử thông báo", color = DarkOnSurface)
                    }
                },
                onClick = {
                    onShowMenuChange(false)
                    onNotificationHistoryClick()
                }
            )

            DropdownMenuItem(
                text = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.SupervisedUserCircle,
                            contentDescription = null,
                            tint = DarkOnSurface,
                            modifier = Modifier.size(20.dp)
                        )
                        Text("Quản lý kết nối", color = DarkOnSurface)
                    }
                },
                onClick = {
                    onShowMenuChange(false)
                    onManageConnectionsClick()
                }
            )

            Divider(color = DarkOnSurface.copy(alpha = 0.2f))

            DropdownMenuItem(
                text = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Call,
                            contentDescription = null,
                            tint = DarkOnSurface,
                            modifier = Modifier.size(20.dp)
                        )
                        Text("Liên hệ khẩn cấp", color = DarkOnSurface)
                    }
                },
                onClick = {
                    onShowMenuChange(false)
                    onEmergencyContactClick()
                }
            )

            Divider(color = DarkOnSurface.copy(alpha = 0.2f))

            DropdownMenuItem(
                text = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.ExitToApp,
                            contentDescription = null,
                            tint = AIError,
                            modifier = Modifier.size(20.dp)
                        )
                        Text("Đăng xuất", color = AIError)
                    }
                },
                onClick = {
                    onShowMenuChange(false)
                    onLogout()
                }
            )
        }
    }
}

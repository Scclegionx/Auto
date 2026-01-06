package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.service.be.NotificationData
import com.auto_fe.auto_fe.service.be.NotificationService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NotificationScreen(
    accessToken: String,
    onBackClick: () -> Unit = {},
    onNotificationClick: (NotificationData) -> Unit = {}
) {
    val scope = rememberCoroutineScope()
    val service = remember { NotificationService() }
    val listState = rememberLazyListState()
    
    var notifications by remember { mutableStateOf<List<NotificationData>>(emptyList()) }
    var unreadCount by remember { mutableStateOf(0L) }
    var isLoading by remember { mutableStateOf(false) }
    var isLoadingMore by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var selectedTab by remember { mutableStateOf(0) } // 0: All, 1: Unread
    var currentPage by remember { mutableStateOf(0) }
    var hasMoreData by remember { mutableStateOf(true) }
    val pageSize = 20
    
    // Load initial data
    fun loadData(page: Int = 0, append: Boolean = false) {
        if (append) {
            isLoadingMore = true
        } else {
            isLoading = true
            errorMessage = null
            currentPage = 0
            hasMoreData = true
            if (!append) notifications = emptyList()
        }
        
        scope.launch {
            try {
                val result = when (selectedTab) {
                    1 -> service.getUnreadNotifications(accessToken, page, pageSize)
                    else -> service.getAllNotifications(accessToken, page, pageSize)
                }
                
                result.onSuccess { data ->
                    if (append) {
                        notifications = notifications + data
                    } else {
                        notifications = data
                    }
                    hasMoreData = data.size >= pageSize
                    currentPage = page
                }.onFailure { error ->
                    errorMessage = error.message
                }
                
                // Load unread count
                service.getUnreadCount(accessToken).onSuccess { count ->
                    unreadCount = count
                }
                
            } catch (e: Exception) {
                errorMessage = e.message
            } finally {
                isLoading = false
                isLoadingMore = false
            }
        }
    }
    
    // Initial load
    LaunchedEffect(selectedTab) {
        loadData()
    }
    
    // Detect scroll to end for pagination
    LaunchedEffect(listState) {
        snapshotFlow { listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index }
            .collect { lastVisibleIndex ->
                if (lastVisibleIndex != null && 
                    lastVisibleIndex >= notifications.size - 3 && 
                    !isLoadingMore && 
                    hasMoreData) {
                    loadData(currentPage + 1, append = true)
                }
            }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            "Thông báo",
                            fontSize = AppTextSize.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = DarkOnPrimary
                        )
                        if (unreadCount > 0) {
                            Text(
                                "$unreadCount chưa đọc",
                                fontSize = AppTextSize.bodySmall,
                                color = DarkOnPrimary.copy(alpha = 0.8f)
                            )
                        }
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Default.ArrowBack, "Quay lại", tint = DarkOnPrimary)
                    }
                },
                actions = {
                    // Mark all as read button
                    if (unreadCount > 0) {
                        IconButton(
                            onClick = {
                                scope.launch {
                                    service.markAllAsRead(accessToken).onSuccess {
                                        unreadCount = 0
                                        // Refresh list
                                        loadData()
                                    }
                                }
                            }
                        ) {
                            Icon(Icons.Default.DoneAll, "Đánh dấu tất cả đã đọc", tint = DarkOnPrimary)
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkPrimary,
                    titleContentColor = DarkOnPrimary
                )
            )
        },
        containerColor = DarkBackground
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Tabs
            TabRow(
                selectedTabIndex = selectedTab,
                containerColor = DarkSurface,
                contentColor = DarkPrimary,
                indicator = { tabPositions ->
                    TabRowDefaults.SecondaryIndicator(
                        Modifier.tabIndicatorOffset(tabPositions[selectedTab]),
                        color = DarkPrimary
                    )
                }
            ) {
                Tab(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    text = {
                        Text(
                            "Tất cả",
                            fontSize = AppTextSize.bodyMedium,
                            fontWeight = if (selectedTab == 0) FontWeight.Bold else FontWeight.Normal
                        )
                    },
                    selectedContentColor = DarkPrimary,
                    unselectedContentColor = DarkOnSurface.copy(alpha = 0.6f)
                )
                
                Tab(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    text = {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                "Chưa đọc",
                                fontSize = AppTextSize.bodyMedium,
                                fontWeight = if (selectedTab == 1) FontWeight.Bold else FontWeight.Normal
                            )
                            if (unreadCount > 0) {
                                Spacer(modifier = Modifier.width(8.dp))
                                Badge(
                                    containerColor = DarkPrimary,
                                    contentColor = DarkOnPrimary
                                ) {
                                    Text(
                                        unreadCount.toString(),
                                        fontSize = AppTextSize.labelSmall
                                    )
                                }
                            }
                        }
                    },
                    selectedContentColor = DarkPrimary,
                    unselectedContentColor = DarkOnSurface.copy(alpha = 0.6f)
                )
            }
            
            // Content
            Box(modifier = Modifier.fillMaxSize()) {
                when {
                    isLoading -> {
                        // Loading state
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                CircularProgressIndicator(color = DarkPrimary)
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    "Đang tải thông báo...",
                                    color = DarkOnSurface.copy(alpha = 0.6f)
                                )
                            }
                        }
                    }
                    
                    errorMessage != null -> {
                        // Error state
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                modifier = Modifier.padding(24.dp)
                            ) {
                                Icon(
                                    Icons.Default.Error,
                                    contentDescription = null,
                                    tint = DarkError,
                                    modifier = Modifier.size(64.dp)
                                )
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    errorMessage ?: "Đã xảy ra lỗi",
                                    color = DarkOnSurface,
                                    fontSize = AppTextSize.bodyMedium
                                )
                                Spacer(modifier = Modifier.height(16.dp))
                                Button(
                                    onClick = { loadData() },
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = DarkPrimary
                                    )
                                ) {
                                    Text("Thử lại")
                                }
                            }
                        }
                    }
                    
                    notifications.isEmpty() -> {
                        // Empty state
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                modifier = Modifier.padding(24.dp)
                            ) {
                                Icon(
                                    Icons.Default.Notifications,
                                    contentDescription = null,
                                    tint = DarkOnSurface.copy(alpha = 0.3f),
                                    modifier = Modifier.size(80.dp)
                                )
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(
                                    if (selectedTab == 1) "Không có thông báo chưa đọc" 
                                    else "Chưa có thông báo nào",
                                    color = DarkOnSurface.copy(alpha = 0.6f),
                                    fontSize = AppTextSize.bodyMedium
                                )
                            }
                        }
                    }
                    
                    else -> {
                        // Notification list
                        LazyColumn(
                            state = listState,
                            modifier = Modifier.fillMaxSize(),
                            contentPadding = PaddingValues(vertical = 8.dp)
                        ) {
                            items(notifications, key = { it.id }) { notification ->
                                NotificationItem(
                                    notification = notification,
                                    service = service,
                                    onClick = {
                                        // Mark as read when clicked
                                        if (!notification.isRead) {
                                            scope.launch {
                                                service.markAsRead(accessToken, notification.id)
                                                    .onSuccess {
                                                        unreadCount = maxOf(0, unreadCount - 1)
                                                        // Update local state
                                                        notifications = notifications.map {
                                                            if (it.id == notification.id) {
                                                                it.copy(isRead = true)
                                                            } else it
                                                        }
                                                    }
                                            }
                                        }
                                        onNotificationClick(notification)
                                    }
                                )
                                Divider(color = DarkOnSurface.copy(alpha = 0.1f))
                            }
                            
                            // Loading more indicator
                            if (isLoadingMore) {
                                item {
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(16.dp),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        CircularProgressIndicator(
                                            color = DarkPrimary,
                                            modifier = Modifier.size(32.dp)
                                        )
                                    }
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
fun NotificationItem(
    notification: NotificationData,
    service: NotificationService,
    onClick: () -> Unit
) {
    val typeIcon = getNotificationIcon(notification.notificationType)
    val typeColor = getNotificationColor(notification.notificationType)
    
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .background(if (!notification.isRead) DarkSurface.copy(alpha = 0.5f) else DarkBackground)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Icon
        Box(
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(typeColor.copy(alpha = 0.1f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                typeIcon,
                contentDescription = null,
                tint = typeColor,
                modifier = Modifier.size(24.dp)
            )
        }
        
        // Content
        Column(
            modifier = Modifier
                .weight(1f)
                .padding(end = 8.dp)
        ) {
            // Type label
            Text(
                service.getNotificationTypeDisplayName(notification.notificationType),
                fontSize = AppTextSize.labelSmall,
                color = typeColor,
                fontWeight = FontWeight.Medium
            )
            
            Spacer(modifier = Modifier.height(4.dp))
            
            // Title
            Text(
                notification.title,
                fontSize = AppTextSize.bodyMedium,
                fontWeight = if (!notification.isRead) FontWeight.Bold else FontWeight.Normal,
                color = DarkOnSurface,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
            
            if (notification.body.isNotEmpty()) {
                Spacer(modifier = Modifier.height(4.dp))
                
                // Body
                Text(
                    notification.body,
                    fontSize = AppTextSize.bodySmall,
                    color = DarkOnSurface.copy(alpha = 0.7f),
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
            
            Spacer(modifier = Modifier.height(4.dp))
            
            // Timestamp
            Text(
                service.formatTimestamp(notification.createdAt),
                fontSize = AppTextSize.labelSmall,
                color = DarkOnSurface.copy(alpha = 0.5f)
            )
        }
        
        // Unread indicator
        if (!notification.isRead) {
            Box(
                modifier = Modifier
                    .size(12.dp)
                    .clip(CircleShape)
                    .background(DarkPrimary)
                    .align(Alignment.Top)
            )
        }
    }
}

/**
 * Get icon for notification type
 */
fun getNotificationIcon(type: String): ImageVector {
    return when (type) {
        "MEDICATION_REMINDER" -> Icons.Default.Notifications
        "ELDER_MISSED_MEDICATION" -> Icons.Default.Warning
        "ELDER_LATE_MEDICATION" -> Icons.Default.Schedule
        "ELDER_ADHERENCE_LOW" -> Icons.Default.TrendingDown
        "ELDER_HEALTH_ALERT" -> Icons.Default.LocalHospital
        "SYSTEM_ANNOUNCEMENT" -> Icons.Default.Campaign
        "RELATIONSHIP_REQUEST" -> Icons.Default.PersonAdd
        "RELATIONSHIP_ACCEPTED" -> Icons.Default.CheckCircle
        else -> Icons.Default.Notifications
    }
}

/**
 * Get color for notification type
 */
fun getNotificationColor(type: String): androidx.compose.ui.graphics.Color {
    return when (type) {
        "MEDICATION_REMINDER" -> DarkPrimary
        "ELDER_MISSED_MEDICATION" -> DarkError
        "ELDER_LATE_MEDICATION" -> androidx.compose.ui.graphics.Color(0xFFFF9800) // Orange
        "ELDER_ADHERENCE_LOW" -> androidx.compose.ui.graphics.Color(0xFFFFC107) // Amber
        "ELDER_HEALTH_ALERT" -> DarkError
        "SYSTEM_ANNOUNCEMENT" -> androidx.compose.ui.graphics.Color(0xFF2196F3) // Blue
        "RELATIONSHIP_REQUEST" -> DarkPrimary
        "RELATIONSHIP_ACCEPTED" -> androidx.compose.ui.graphics.Color(0xFF4CAF50) // Green
        else -> DarkOnSurface
    }
}

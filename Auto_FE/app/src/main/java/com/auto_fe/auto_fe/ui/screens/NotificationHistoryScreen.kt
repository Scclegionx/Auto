package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.service.NotificationHistoryResponse
import com.auto_fe.auto_fe.service.NotificationHistoryService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NotificationHistoryScreen(
    accessToken: String,
    onBack: () -> Unit
) {
    val scope = rememberCoroutineScope()
    val service = remember { NotificationHistoryService() }
    
    var notifications by remember { mutableStateOf<List<NotificationHistoryResponse>>(emptyList()) }
    var unreadCount by remember { mutableStateOf(0L) }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var selectedFilter by remember { mutableStateOf("all") } // all, today, week, unread
    
    // Load data
    LaunchedEffect(selectedFilter) {
        isLoading = true
        errorMessage = null
        
        scope.launch {
            try {
                val result = when (selectedFilter) {
                    "today" -> service.getTodayHistory(accessToken)
                    "week" -> service.getWeekHistory(accessToken)
                    "unread" -> service.getHistory(accessToken, limit = 50)
                        .map { list -> list.filter { !it.isRead } }
                    else -> service.getHistory(accessToken, limit = 50)
                }
                
                result.onSuccess { data ->
                    notifications = data
                    isLoading = false
                }.onFailure { error ->
                    errorMessage = error.message
                    isLoading = false
                }
                
                // Load unread count
                service.getUnreadCount(accessToken).onSuccess { count ->
                    unreadCount = count
                }
                
            } catch (e: Exception) {
                errorMessage = e.message
                isLoading = false
            }
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Column {
                        Text("Lịch sử thông báo")
                        if (unreadCount > 0) {
                            Text(
                                "$unreadCount chưa đọc",
                                fontSize = 12.sp,
                                color = Color.Gray
                            )
                        }
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, "Quay lại")
                    }
                },
                actions = {
                    if (unreadCount > 0) {
                        TextButton(
                            onClick = {
                                scope.launch {
                                    service.markAllAsRead(accessToken).onSuccess {
                                        // Reload
                                        selectedFilter = "all"
                                    }
                                }
                            }
                        ) {
                            Text("Đánh dấu tất cả đã đọc")
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkPrimary,
                    titleContentColor = DarkOnPrimary
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .background(DarkBackground)
        ) {
            // Filter tabs
            ScrollableTabRow(
                selectedTabIndex = when (selectedFilter) {
                    "all" -> 0
                    "today" -> 1
                    "week" -> 2
                    "unread" -> 3
                    else -> 0
                },
                containerColor = DarkSurface,
                contentColor = DarkOnSurface
            ) {
                Tab(
                    selected = selectedFilter == "all",
                    onClick = { selectedFilter = "all" },
                    text = { Text("Tất cả") }
                )
                Tab(
                    selected = selectedFilter == "today",
                    onClick = { selectedFilter = "today" },
                    text = { Text("Hôm nay") }
                )
                Tab(
                    selected = selectedFilter == "week",
                    onClick = { selectedFilter = "week" },
                    text = { Text("7 ngày") }
                )
                Tab(
                    selected = selectedFilter == "unread",
                    onClick = { selectedFilter = "unread" },
                    text = { 
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Chưa đọc")
                            if (unreadCount > 0) {
                                Spacer(Modifier.width(4.dp))
                                Badge { Text("$unreadCount") }
                            }
                        }
                    }
                )
            }
            
            // Content
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
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(
                                Icons.Default.Warning,
                                contentDescription = null,
                                modifier = Modifier.size(64.dp),
                                tint = Color.Red
                            )
                            Spacer(Modifier.height(8.dp))
                            Text(
                                errorMessage ?: "Lỗi không xác định",
                                color = DarkOnBackground
                            )
                            Spacer(Modifier.height(16.dp))
                            Button(onClick = { selectedFilter = "all" }) {
                                Text("Thử lại")
                            }
                        }
                    }
                }
                
                notifications.isEmpty() -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(
                                Icons.Default.Notifications,
                                contentDescription = null,
                                modifier = Modifier.size(64.dp),
                                tint = Color.Gray
                            )
                            Spacer(Modifier.height(8.dp))
                            Text(
                                "Không có thông báo",
                                color = Color.Gray
                            )
                        }
                    }
                }
                
                else -> {
                    LazyColumn(
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(notifications) { notification ->
                            NotificationHistoryItem(
                                notification = notification,
                                onMarkAsRead = {
                                    scope.launch {
                                        service.markAsRead(accessToken, notification.id).onSuccess {
                                            // Reload
                                            selectedFilter = selectedFilter
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
fun NotificationHistoryItem(
    notification: NotificationHistoryResponse,
    onMarkAsRead: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { if (!notification.isRead) onMarkAsRead() },
        colors = CardDefaults.cardColors(
            containerColor = if (notification.isRead) DarkSurface else DarkPrimary.copy(alpha = 0.1f)
        ),
        shape = RoundedCornerShape(12.dp)
    ) {
        Row(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            // Unread indicator
            if (!notification.isRead) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(DarkPrimary)
                        .align(Alignment.Top)
                )
                Spacer(Modifier.width(12.dp))
            }
            
            // Content
            Column(modifier = Modifier.weight(1f)) {
                // Title
                Text(
                    text = notification.title ?: "Thông báo uống thuốc",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface
                )
                
                Spacer(Modifier.height(4.dp))
                
                // Medication names
                Text(
                    text = notification.medicationNames ?: "",
                    fontSize = 14.sp,
                    color = DarkOnSurface.copy(alpha = 0.8f)
                )
                
                Spacer(Modifier.height(8.dp))
                
                // Time and status
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Time
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = "🕐",
                            fontSize = 16.sp,
                            color = Color.Gray
                        )
                        Spacer(Modifier.width(4.dp))
                        Text(
                            text = formatDateTime(notification.reminderTime),
                            fontSize = 12.sp,
                            color = Color.Gray
                        )
                    }
                    
                    // Status
                    val statusColor = when (notification.status) {
                        "SENT" -> Color.Green
                        "FAILED" -> Color.Red
                        else -> Color.Gray
                    }
                    val statusText = when (notification.status) {
                        "SENT" -> "Đã gửi"
                        "FAILED" -> "Thất bại"
                        else -> notification.status
                    }
                    
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(statusColor)
                        )
                        Spacer(Modifier.width(4.dp))
                        Text(
                            text = statusText,
                            fontSize = 12.sp,
                            color = statusColor
                        )
                    }
                    
                    // Medication count
                    if (notification.medicationCount > 1) {
                        Badge {
                            Text("${notification.medicationCount} thuốc")
                        }
                    }
                }
            }
        }
    }
}

fun formatDateTime(dateTimeStr: String): String {
    return try {
        val dateTime = LocalDateTime.parse(dateTimeStr)
        val formatter = DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm")
        dateTime.format(formatter)
    } catch (e: Exception) {
        dateTimeStr
    }
}

package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.auto_fe.auto_fe.service.be.RelationshipService
import com.auto_fe.auto_fe.service.be.UserService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

/**
 * Màn hình quản lý kết nối
 * - Elder: Hiển thị danh sách Supervisor đã kết nối và yêu cầu pending
 * - Supervisor: Hiển thị yêu cầu kết nối từ Elder (pending)
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ManageConnectionsScreen(
    accessToken: String,
    userRole: String = "ELDER", // ELDER hoặc SUPERVISOR
    onBackClick: () -> Unit = {},
    onSearchUserClick: () -> Unit = {}
) {
    val scope = rememberCoroutineScope()
    val relationshipService = remember { RelationshipService() }
    val userService = remember { UserService() }
    
    var connectedSupervisors by remember { mutableStateOf<List<RelationshipService.RelationshipRequest>>(emptyList()) }
    var pendingRequests by remember { mutableStateOf<List<RelationshipService.RelationshipRequest>>(emptyList()) }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showAcceptDialog by remember { mutableStateOf<RelationshipService.RelationshipRequest?>(null) }
    var showRejectDialog by remember { mutableStateOf<RelationshipService.RelationshipRequest?>(null) }
    var selectedTab by remember { mutableStateOf(0) }
    
    // Load dữ liệu dựa trên role
    fun loadData() {
        isLoading = true
        scope.launch {
            try {
                if (userRole == "ELDER") {
                    // ELDER: Load Supervisors đã kết nối
                    relationshipService.getConnectedSupervisors(accessToken).onSuccess { supervisors ->
                        connectedSupervisors = supervisors
                    }
                } else {
                    // SUPERVISOR: Load Elders đã kết nối (không cần hiện ở đây, vì có ElderListScreen rồi)
                    connectedSupervisors = emptyList()
                }
                
                // Load yêu cầu pending (cả Elder và Supervisor đều có)
                relationshipService.getPendingReceivedRequests(accessToken).onSuccess { requests ->
                    pendingRequests = requests
                }
                
                isLoading = false
            } catch (e: Exception) {
                errorMessage = e.message
                isLoading = false
            }
        }
    }
    
    LaunchedEffect(Unit) {
        loadData()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        if (userRole == "ELDER") "Quản lý kết nối" else "Quản lý yêu cầu",
                        fontSize = AppTextSize.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnPrimary
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription = "Quay lại",
                            tint = DarkOnPrimary
                        )
                    }
                },
                actions = {
                    // Chỉ Elder mới có nút thêm Supervisor
                    if (userRole == "ELDER") {
                        IconButton(onClick = onSearchUserClick) {
                            Icon(
                                Icons.Default.PersonAdd,
                                contentDescription = "Thêm người giám sát",
                                tint = DarkOnPrimary
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkPrimary
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Tabs (Supervisor chỉ có tab Pending)
            if (userRole == "ELDER") {
                TabRow(
                    selectedTabIndex = selectedTab,
                    containerColor = DarkSurface,
                    contentColor = DarkOnPrimary
                ) {
                    Tab(
                        selected = selectedTab == 0,
                        onClick = { selectedTab = 0 },
                        text = {
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text("Đã kết nối")
                                if (connectedSupervisors.isNotEmpty()) {
                                    Badge {
                                        Text("${connectedSupervisors.size}")
                                    }
                                }
                            }
                        }
                    )
                    Tab(
                        selected = selectedTab == 1,
                        onClick = { selectedTab = 1 },
                        text = {
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text("Yêu cầu")
                                if (pendingRequests.isNotEmpty()) {
                                    Badge(
                                        containerColor = DarkError
                                    ) {
                                        Text("${pendingRequests.size}")
                                    }
                                }
                            }
                        }
                    )
                }
            } else {
                // SUPERVISOR: Chỉ hiện "Yêu cầu kết nối"
                Text(
                    "Yêu cầu kết nối (${pendingRequests.size})",
                    fontSize = AppTextSize.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface,
                    modifier = Modifier.padding(16.dp)
                )
            }

            // Content
            if (isLoading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator()
                }
            } else {
                if (userRole == "ELDER") {
                    // ELDER: Hiển thị tabs (Connected/Pending)
                    when (selectedTab) {
                        0 -> ConnectedSupervisorsTab(
                            supervisors = connectedSupervisors,
                            onSearchClick = onSearchUserClick,
                            accessToken = accessToken,
                            onPermissionsUpdated = { loadData() } // Reload sau khi cập nhật
                        )
                        1 -> PendingRequestsTab(
                            requests = pendingRequests,
                            onAccept = { request -> showAcceptDialog = request },
                            onReject = { request -> showRejectDialog = request },
                            userRole = userRole
                        )
                    }
                } else {
                    // SUPERVISOR: Chỉ hiển thị Pending requests
                    PendingRequestsTab(
                        requests = pendingRequests,
                        onAccept = { request -> showAcceptDialog = request },
                        onReject = { request -> showRejectDialog = request },
                        userRole = userRole
                    )
                }
            }
        }
    }

    // Accept Dialog
    showAcceptDialog?.let { request ->
        var responseMessage by remember { mutableStateOf("") }
        
        AlertDialog(
            onDismissRequest = { showAcceptDialog = null },
            title = { Text("Chấp nhận yêu cầu") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Bạn có chắc muốn chấp nhận yêu cầu kết nối từ ${request.requesterName}?")
                    OutlinedTextField(
                        value = responseMessage,
                        onValueChange = { responseMessage = it },
                        label = { Text("Lời nhắn (tùy chọn)") },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        scope.launch {
                            relationshipService.acceptRequest(
                                accessToken,
                                request.id,
                                responseMessage.ifBlank { null }
                            ).onSuccess {
                                showAcceptDialog = null
                                loadData() // Reload
                            }.onFailure { error ->
                                errorMessage = error.message
                            }
                        }
                    }
                ) {
                    Text("Chấp nhận")
                }
            },
            dismissButton = {
                TextButton(onClick = { showAcceptDialog = null }) {
                    Text("Hủy")
                }
            }
        )
    }

    // Reject Dialog
    showRejectDialog?.let { request ->
        var responseMessage by remember { mutableStateOf("") }
        
        AlertDialog(
            onDismissRequest = { showRejectDialog = null },
            title = { Text("Từ chối yêu cầu") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Bạn có chắc muốn từ chối yêu cầu kết nối từ ${request.requesterName}?")
                    OutlinedTextField(
                        value = responseMessage,
                        onValueChange = { responseMessage = it },
                        label = { Text("Lý do (tùy chọn)") },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        scope.launch {
                            relationshipService.rejectRequest(
                                accessToken,
                                request.id,
                                responseMessage.ifBlank { null }
                            ).onSuccess {
                                showRejectDialog = null
                                loadData() // Reload
                            }.onFailure { error ->
                                errorMessage = error.message
                            }
                        }
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = DarkError
                    )
                ) {
                    Text("Từ chối")
                }
            },
            dismissButton = {
                TextButton(onClick = { showRejectDialog = null }) {
                    Text("Hủy")
                }
            }
        )
    }

    // Error Snackbar
    errorMessage?.let { error ->
        LaunchedEffect(error) {
            // Show snackbar
            errorMessage = null
        }
    }
}

@Composable
fun ConnectedSupervisorsTab(
    supervisors: List<RelationshipService.RelationshipRequest>,
    onSearchClick: () -> Unit,
    accessToken: String,
    onPermissionsUpdated: () -> Unit = {}
) {
    if (supervisors.isEmpty()) {
        // Empty state
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Icon(
                    Icons.Default.SupervisedUserCircle,
                    contentDescription = null,
                    modifier = Modifier.size(64.dp),
                    tint = DarkOnSurface.copy(alpha = 0.5f)
                )
                Text(
                    "Chưa có người giám sát",
                    fontSize = AppTextSize.bodyLarge,
                    color = DarkOnSurface.copy(alpha = 0.7f)
                )
                Button(onClick = onSearchClick) {
                    Icon(Icons.Default.PersonAdd, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Thêm người giám sát")
                }
            }
        }
    } else {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(supervisors) { relationship ->
                SupervisorCard(
                    relationship = relationship,
                    accessToken = accessToken,
                    onPermissionsUpdated = onPermissionsUpdated
                )
            }
        }
    }
}

@Composable
fun SupervisorCard(
    relationship: RelationshipService.RelationshipRequest,
    accessToken: String,
    onPermissionsUpdated: () -> Unit = {}
) {
    val scope = rememberCoroutineScope()
    val relationshipService = remember { RelationshipService() }
    var showEditPermissionsDialog by remember { mutableStateOf(false) }
    var isUpdating by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.Top
        ) {
            // Avatar
            Box(
                modifier = Modifier
                    .size(64.dp)
                    .clip(CircleShape)
                    .background(DarkPrimary.copy(0.1f)),
                contentAlignment = Alignment.Center
            ) {
                if (!relationship.supervisorUserAvatar.isNullOrBlank()) {
                    AsyncImage(
                        model = relationship.supervisorUserAvatar,
                        contentDescription = "Avatar",
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(CircleShape)
                    )
                } else {
                    Text(
                        relationship.supervisorUserName.firstOrNull()?.toString()?.uppercase() ?: "?",
                        fontSize = AppTextSize.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = DarkPrimary
                    )
                }
            }

            // Info
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                // Name
                Text(
                    relationship.supervisorUserName,
                    fontSize = AppTextSize.bodyLarge,
                    fontWeight = FontWeight.Bold,
                    color = DarkOnSurface,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                
                // Email
                if (!relationship.supervisorUserEmail.isNullOrBlank()) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Icon(
                            Icons.Default.Email,
                            contentDescription = null,
                            tint = DarkOnSurface.copy(alpha = 0.6f),
                            modifier = Modifier.size(16.dp)
                        )
                        Text(
                            relationship.supervisorUserEmail,
                            fontSize = AppTextSize.bodySmall,
                            color = DarkOnSurface.copy(alpha = 0.7f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
                
                // Occupation & Workplace chips
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.padding(top = 4.dp)
                ) {
                    relationship.supervisorOccupation?.let { occupation ->
                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            color = DarkPrimary.copy(alpha = 0.15f)
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                                horizontalArrangement = Arrangement.spacedBy(4.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    Icons.Default.Work,
                                    contentDescription = null,
                                    tint = DarkPrimary,
                                    modifier = Modifier.size(14.dp)
                                )
                                Text(
                                    occupation,
                                    fontSize = AppTextSize.labelSmall,
                                    color = DarkPrimary,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }
                    }
                    
                    relationship.supervisorWorkplace?.let { workplace ->
                        Surface(
                            shape = RoundedCornerShape(8.dp),
                            color = Color(0xFF2196F3).copy(alpha = 0.15f)
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                                horizontalArrangement = Arrangement.spacedBy(4.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    Icons.Default.LocationOn,
                                    contentDescription = null,
                                    tint = Color(0xFF2196F3),
                                    modifier = Modifier.size(14.dp)
                                )
                                Text(
                                    workplace,
                                    fontSize = AppTextSize.labelSmall,
                                    color = Color(0xFF2196F3),
                                    fontWeight = FontWeight.Medium,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    }
                }
                
                // Status badge
                Spacer(modifier = Modifier.height(4.dp))
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = Color(0xFF4CAF50).copy(alpha = 0.15f)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.CheckCircle,
                            contentDescription = null,
                            tint = Color(0xFF4CAF50),
                            modifier = Modifier.size(16.dp)
                        )
                        Text(
                            "Đang giám sát",
                            fontSize = AppTextSize.labelSmall,
                            color = Color(0xFF4CAF50),
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
                
                // Permission badges
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.padding(top = 4.dp)
                ) {
                    // View permission badge
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = if (relationship.canViewMedications) 
                            Color(0xFF2196F3).copy(alpha = 0.15f) 
                        else 
                            DarkError.copy(alpha = 0.15f)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            horizontalArrangement = Arrangement.spacedBy(4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                if (relationship.canViewMedications) Icons.Default.Visibility else Icons.Default.VisibilityOff,
                                contentDescription = null,
                                tint = if (relationship.canViewMedications) Color(0xFF2196F3) else DarkError,
                                modifier = Modifier.size(14.dp)
                            )
                            Text(
                                if (relationship.canViewMedications) "Xem thuốc" else "Không xem",
                                fontSize = AppTextSize.labelSmall,
                                color = if (relationship.canViewMedications) Color(0xFF2196F3) else DarkError,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                    
                    // Update permission badge
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = if (relationship.canUpdateMedications) 
                            Color(0xFF4CAF50).copy(alpha = 0.15f) 
                        else 
                            DarkError.copy(alpha = 0.15f)
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            horizontalArrangement = Arrangement.spacedBy(4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                if (relationship.canUpdateMedications) Icons.Default.Edit else Icons.Default.Block,
                                contentDescription = null,
                                tint = if (relationship.canUpdateMedications) Color(0xFF4CAF50) else DarkError,
                                modifier = Modifier.size(14.dp)
                            )
                            Text(
                                if (relationship.canUpdateMedications) "Chỉnh sửa" else "Không sửa",
                                fontSize = AppTextSize.labelSmall,
                                color = if (relationship.canUpdateMedications) Color(0xFF4CAF50) else DarkError,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                }
                
                // Edit permissions button
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedButton(
                    onClick = { showEditPermissionsDialog = true },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = DarkPrimary
                    )
                ) {
                    Icon(
                        Icons.Default.Settings,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(Modifier.width(8.dp))
                    Text("Chỉnh sửa quyền", fontSize = AppTextSize.bodySmall)
                }
            }
        }
    }
    
    // Edit Permissions Dialog
    if (showEditPermissionsDialog) {
        var canView by remember { mutableStateOf(relationship.canViewMedications) }
        var canUpdate by remember { mutableStateOf(relationship.canUpdateMedications) }
        
        AlertDialog(
            onDismissRequest = { showEditPermissionsDialog = false },
            title = { 
                Text(
                    "Chỉnh sửa quyền",
                    fontSize = AppTextSize.titleMedium,
                    fontWeight = FontWeight.Bold
                ) 
            },
            text = {
                Column(
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Text(
                        "Cập nhật quyền truy cập của ${relationship.supervisorUserName}",
                        fontSize = AppTextSize.bodyMedium,
                        color = DarkOnSurface.copy(alpha = 0.8f)
                    )
                    
                    HorizontalDivider()
                    
                    // Quyền xem thuốc
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Xem thông tin thuốc",
                                fontSize = AppTextSize.bodyMedium,
                                fontWeight = FontWeight.Medium,
                                color = DarkOnSurface
                            )
                            Text(
                                "Cho phép xem đơn thuốc và lịch sử uống thuốc",
                                fontSize = AppTextSize.bodySmall,
                                color = DarkOnSurface.copy(alpha = 0.6f)
                            )
                        }
                        Switch(
                            checked = canView,
                            onCheckedChange = { canView = it }
                        )
                    }
                    
                    HorizontalDivider()
                    
                    // Quyền chỉnh sửa thuốc
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                "Chỉnh sửa thuốc",
                                fontSize = AppTextSize.bodyMedium,
                                fontWeight = FontWeight.Medium,
                                color = DarkOnSurface
                            )
                            Text(
                                "Cho phép thêm, sửa, xóa đơn thuốc",
                                fontSize = AppTextSize.bodySmall,
                                color = DarkOnSurface.copy(alpha = 0.6f)
                            )
                        }
                        Switch(
                            checked = canUpdate,
                            onCheckedChange = { canUpdate = it }
                        )
                    }
                    
                    // Warning nếu tắt quyền xem
                    if (!canView) {
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = DarkError.copy(alpha = 0.1f)
                            ),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Icon(
                                    Icons.Default.Warning,
                                    contentDescription = null,
                                    tint = DarkError,
                                    modifier = Modifier.size(20.dp)
                                )
                                Text(
                                    "Tắt quyền xem sẽ tự động tắt quyền chỉnh sửa",
                                    fontSize = AppTextSize.bodySmall,
                                    color = DarkError
                                )
                            }
                        }
                    }
                    
                    // Error message
                    errorMessage?.let { error ->
                        Text(
                            error,
                            fontSize = AppTextSize.bodySmall,
                            color = DarkError
                        )
                    }
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        isUpdating = true
                        errorMessage = null
                        
                        // Nếu tắt quyền xem thì tự động tắt quyền sửa
                        val finalCanUpdate = if (!canView) false else canUpdate
                        
                        scope.launch {
                            relationshipService.updatePermissions(
                                accessToken = accessToken,
                                supervisorId = relationship.supervisorUserId,
                                canViewMedications = canView,
                                canUpdateMedications = finalCanUpdate
                            ).fold(
                                onSuccess = {
                                    showEditPermissionsDialog = false
                                    isUpdating = false
                                    onPermissionsUpdated() // Reload danh sách
                                },
                                onFailure = { error ->
                                    errorMessage = error.message
                                    isUpdating = false
                                }
                            )
                        }
                    },
                    enabled = !isUpdating
                ) {
                    if (isUpdating) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            color = DarkOnPrimary
                        )
                    } else {
                        Text("Lưu")
                    }
                }
            },
            dismissButton = {
                TextButton(
                    onClick = { showEditPermissionsDialog = false },
                    enabled = !isUpdating
                ) {
                    Text("Hủy")
                }
            }
        )
    }
}

@Composable
fun PendingRequestsTab(
    requests: List<RelationshipService.RelationshipRequest>,
    onAccept: (RelationshipService.RelationshipRequest) -> Unit,
    onReject: (RelationshipService.RelationshipRequest) -> Unit,
    userRole: String = "ELDER"
) {
    if (requests.isEmpty()) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Icon(
                    Icons.Default.Notifications,
                    contentDescription = null,
                    modifier = Modifier.size(64.dp),
                    tint = DarkOnSurface.copy(alpha = 0.5f)
                )
                Text(
                    "Không có yêu cầu mới",
                    fontSize = AppTextSize.bodyLarge,
                    color = DarkOnSurface.copy(alpha = 0.7f)
                )
            }
        }
    } else {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(requests) { request ->
                PendingRequestCard(
                    request = request,
                    onAccept = { onAccept(request) },
                    onReject = { onReject(request) }
                )
            }
        }
    }
}

@Composable
fun PendingRequestCard(
    request: RelationshipService.RelationshipRequest,
    onAccept: () -> Unit,
    onReject: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Header
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(48.dp)
                        .clip(CircleShape)
                        .background(DarkPrimary),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        request.requesterName.firstOrNull()?.toString() ?: "?",
                        fontSize = AppTextSize.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnPrimary
                    )
                }

                Column(
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        request.requesterName,
                        fontSize = AppTextSize.bodyLarge,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                    Text(
                        "Yêu cầu giám sát",
                        fontSize = AppTextSize.bodySmall,
                        color = DarkOnSurface.copy(alpha = 0.7f)
                    )
                }
            }

            // Message
            if (!request.requestMessage.isNullOrBlank()) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = DarkBackground
                    )
                ) {
                    Text(
                        request.requestMessage,
                        fontSize = AppTextSize.bodyMedium,
                        color = DarkOnSurface,
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }

            // Actions
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = onReject,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = DarkError
                    )
                ) {
                    Icon(Icons.Default.Close, contentDescription = null)
                    Spacer(Modifier.width(4.dp))
                    Text("Từ chối")
                }
                Button(
                    onClick = onAccept,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.Check, contentDescription = null)
                    Spacer(Modifier.width(4.dp))
                    Text("Chấp nhận")
                }
            }
        }
    }
}

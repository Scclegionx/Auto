package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.CreateMedicationRequest;
import com.example.Auto_BE.dto.request.UpdateMedicationRequest;
import com.example.Auto_BE.dto.response.MedicationResponse;
import com.example.Auto_BE.entity.ElderSupervisor;
import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.MedicationReminder;
import com.example.Auto_BE.entity.Prescriptions;
import com.example.Auto_BE.entity.SupervisorUser;
import com.example.Auto_BE.entity.enums.ETypeMedication;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.repository.ElderSupervisorRepository;
import com.example.Auto_BE.repository.ElderUserRepository;
import com.example.Auto_BE.repository.MedicationReminderRepository;
import com.example.Auto_BE.repository.PrescriptionRepository;
import com.example.Auto_BE.repository.SupervisorUserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.core.Authentication;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.*;

/**
 * Unit Test cho MedicationService
 * Sử dụng Mockito để mock các dependencies
 */
@ExtendWith(MockitoExtension.class)
class MedicationServiceTest {

    @Mock
    private MedicationReminderRepository medicationReminderRepository;

    @Mock
    private ElderUserRepository elderUserRepository;

    @Mock
    private SupervisorUserRepository supervisorUserRepository;

    @Mock
    private PrescriptionRepository prescriptionRepository;

    @Mock
    private SimpleTimeBasedScheduler simpleTimeBasedScheduler;

    @Mock
    private ElderSupervisorRepository elderSupervisorRepository;

    @Mock
    private Authentication authentication;

    @InjectMocks
    private MedicationService medicationService;

    private ElderUser testElderUser;
    private SupervisorUser testSupervisorUser;
    private CreateMedicationRequest createRequest;
    private MedicationReminder testMedication;

    @BeforeEach
    void setUp() {
        // Setup test data
        testElderUser = new ElderUser();
        testElderUser.setId(1L);
        testElderUser.setEmail("elder@test.com");
        testElderUser.setFullName("Test Elder");

        testSupervisorUser = new SupervisorUser();
        testSupervisorUser.setId(2L);
        testSupervisorUser.setEmail("supervisor@test.com");
        testSupervisorUser.setFullName("Test Supervisor");

        createRequest = new CreateMedicationRequest();
        createRequest.setName("Paracetamol");
        createRequest.setDescription("Uống sau bữa ăn");
        createRequest.setType(ETypeMedication.OVER_THE_COUNTER);
        createRequest.setReminderTimes(Arrays.asList("08:00", "20:00"));
        createRequest.setDaysOfWeek("1001011");
        createRequest.setIsActive(true);

        testMedication = new MedicationReminder();
        testMedication.setId(1L);
        testMedication.setName("Paracetamol");
        testMedication.setDescription("Uống sau bữa ăn");
        testMedication.setType(ETypeMedication.OVER_THE_COUNTER);
        testMedication.setReminderTime("08:00");
        testMedication.setDaysOfWeek("1001011");
        testMedication.setIsActive(true);
        testMedication.setElderUser(testElderUser);
    }

    /**
     * Test 1: createMedication - Elder tự tạo cho mình (SUCCESS)
     * Scenario: Elder user tạo medication reminder cho chính mình
     * Expected: Tạo thành công 2 medications (1 cho mỗi reminder time)
     */
    @Test
    void testCreateMedication_ElderCreateForSelf_Success() {
        // Arrange
        createRequest.setElderUserId(null); // Elder tự tạo
        when(authentication.getName()).thenReturn("elder@test.com");
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        
        // Mock save để return medication với ID
        when(medicationReminderRepository.save(any(MedicationReminder.class)))
                .thenAnswer(invocation -> {
                    MedicationReminder med = invocation.getArgument(0);
                    med.setId(System.currentTimeMillis()); // Unique ID
                    return med;
                });

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(createRequest, authentication);

        // Assert
        assertNotNull(response);
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
        assertEquals(2, response.getData().size()); // 2 reminder times = 2 medications
        assertEquals("Tạo thành công 2 medications", response.getMessage());
        
        // Verify repository calls
        verify(elderUserRepository, times(1)).findByEmail("elder@test.com");
        verify(medicationReminderRepository, times(2)).save(any(MedicationReminder.class));
        verify(simpleTimeBasedScheduler, times(1)).scheduleUserReminders(testElderUser.getId());
        
        // Verify medication details
        MedicationResponse firstMed = response.getData().get(0);
        assertEquals("Paracetamol", firstMed.getMedicationName());
        assertEquals("Uống sau bữa ăn", firstMed.getDescription());
        assertEquals(ETypeMedication.OVER_THE_COUNTER.name(), firstMed.getType());
    }

    /**
     * Test 2: createMedication - Supervisor tạo cho Elder (SUCCESS)
     * Scenario: Supervisor có quyền tạo medication cho Elder
     * Expected: Tạo thành công medications
     */
    @Test
    void testCreateMedication_SupervisorCreateForElder_Success() {
        // Arrange
        createRequest.setElderUserId(1L); // Supervisor tạo cho Elder ID 1
        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(elderUserRepository.findById(1L)).thenReturn(Optional.of(testElderUser));
        when(supervisorUserRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        
        // Mock ElderSupervisor relationship với quyền update
        ElderSupervisor elderSupervisor = new ElderSupervisor();
        when(elderSupervisorRepository.findActiveWithUpdatePermission(2L, 1L))
                .thenReturn(Optional.of(elderSupervisor)); // Has permission
        
        when(medicationReminderRepository.save(any(MedicationReminder.class)))
                .thenAnswer(invocation -> {
                    MedicationReminder med = invocation.getArgument(0);
                    med.setId(System.currentTimeMillis());
                    return med;
                });

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(createRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals(2, response.getData().size());
        
        verify(elderUserRepository, times(1)).findById(1L);
        verify(supervisorUserRepository, times(1)).findByEmail("supervisor@test.com");
        verify(elderSupervisorRepository, times(1)).findActiveWithUpdatePermission(2L, 1L);
        verify(medicationReminderRepository, times(2)).save(any(MedicationReminder.class));
    }

    /**
     * Test 3: createMedication - Supervisor không có quyền (FAIL)
     * Scenario: Supervisor không có update permission cho Elder
     * Expected: Return error response (service catches exception internally)
     */
    @Test
    void testCreateMedication_SupervisorNoPermission_ThrowsException() {
        // Arrange
        createRequest.setElderUserId(1L);
        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(elderUserRepository.findById(1L)).thenReturn(Optional.of(testElderUser));
        when(supervisorUserRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        when(elderSupervisorRepository.findActiveWithUpdatePermission(2L, 1L))
                .thenReturn(Optional.empty()); // No permission

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(createRequest, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Không có quyền tạo medication cho Elder này"));
        assertNull(response.getData());
        verify(medicationReminderRepository, never()).save(any(MedicationReminder.class));
    }

    /**
     * Test 4: createMedication - Elder không tồn tại (FAIL)
     * Scenario: ElderUserId trong request không tồn tại
     * Expected: Return error response (service catches exception internally)
     */
    @Test
    void testCreateMedication_ElderNotFound_ThrowsException() {
        // Arrange
        createRequest.setElderUserId(999L); // Non-existent ID
        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(elderUserRepository.findById(999L)).thenReturn(Optional.empty());

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(createRequest, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Elder user không tồn tại"));
        assertNull(response.getData());
        verify(medicationReminderRepository, never()).save(any(MedicationReminder.class));
    }

    /**
     * Test 5: getMedicationById - Medication tồn tại (SUCCESS)
     * Scenario: Tìm medication theo ID hợp lệ
     * Expected: Trả về medication details
     */
    @Test
    void testGetMedicationById_ExistingId_Success() {
        // Arrange
        Long medicationId = 1L;
        when(medicationReminderRepository.findById(medicationId)).thenReturn(Optional.of(testMedication));

        // Act
        BaseResponse<MedicationResponse> response = medicationService.getMedicationById(medicationId, authentication);

        // Assert
        assertNotNull(response);
        assertEquals("success", response.getStatus());
        assertEquals("Lấy medication thành công", response.getMessage());
        assertNotNull(response.getData());
        
        MedicationResponse data = response.getData();
        assertEquals("Paracetamol", data.getMedicationName());
        assertEquals("Uống sau bữa ăn", data.getDescription());
        assertEquals("08:00", data.getReminderTimes());
        assertEquals(ETypeMedication.OVER_THE_COUNTER.name(), data.getType());
        assertTrue(data.getIsActive());
        
        verify(medicationReminderRepository, times(1)).findById(medicationId);
    }

    /**
     * Test 6: getMedicationById - Medication không tồn tại (FAIL)
     * Scenario: ID không tồn tại trong database
     * Expected: Trả về response lỗi
     */
    @Test
    void testGetMedicationById_NonExistingId_ReturnsError() {
        // Arrange
        Long nonExistentId = 999L;
        when(medicationReminderRepository.findById(nonExistentId)).thenReturn(Optional.empty());

        // Act
        BaseResponse<MedicationResponse> response = medicationService.getMedicationById(nonExistentId, authentication);

        // Assert
        assertNotNull(response);
        assertEquals("error", response.getStatus());
        assertEquals("Medication không tồn tại", response.getMessage());
        assertNull(response.getData());
        
        verify(medicationReminderRepository, times(1)).findById(nonExistentId);
    }

    /**
     * Test 7: createMedication - Empty reminder times (EDGE CASE)
     * Scenario: Request không có reminder times
     * Expected: Không tạo medication nào, return empty list
     */
    @Test
    void testCreateMedication_EmptyReminderTimes_ReturnsEmptyList() {
        // Arrange
        createRequest.setReminderTimes(Arrays.asList()); // Empty list
        when(authentication.getName()).thenReturn("elder@test.com");
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(createRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertTrue(response.getData().isEmpty());
        assertEquals("Tạo thành công 0 medications", response.getMessage());
        
        verify(medicationReminderRepository, never()).save(any(MedicationReminder.class));
    }

    /**
     * Test 8: createMedication - Scheduler fails but medication created (PARTIAL SUCCESS)
     * Scenario: Medication được tạo nhưng scheduler throw exception
     * Expected: Medication vẫn được tạo, log error nhưng không throw exception
     */
    @Test
    void testCreateMedication_SchedulerFails_MedicationStillCreated() throws Exception {
        // Arrange
        createRequest.setElderUserId(null);
        when(authentication.getName()).thenReturn("elder@test.com");
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class)))
                .thenAnswer(invocation -> {
                    MedicationReminder med = invocation.getArgument(0);
                    med.setId(System.currentTimeMillis());
                    return med;
                });
        
        // Scheduler throws exception
        doThrow(new RuntimeException("Scheduling failed")).when(simpleTimeBasedScheduler).scheduleUserReminders(anyLong());

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(createRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus()); // Vẫn success vì medication đã được tạo
        assertEquals(2, response.getData().size());
        
        verify(medicationReminderRepository, times(2)).save(any(MedicationReminder.class));
        verify(simpleTimeBasedScheduler, times(1)).scheduleUserReminders(testElderUser.getId());
    }

    // ==================== getAllMedicationsByUser Tests ====================

    /**
     * Test 9: getAllMedicationsByUser - Success
     * Scenario: Get all medications for a user
     * Expected: Return list of medications
     */
    @Test
    void testGetAllMedicationsByUser_Success() {
        // Arrange
        MedicationReminder med1 = new MedicationReminder();
        med1.setId(1L);
        med1.setName("Med1");
        med1.setReminderTime("08:00");
        med1.setElderUser(testElderUser);

        MedicationReminder med2 = new MedicationReminder();
        med2.setId(2L);
        med2.setName("Med2");
        med2.setReminderTime("20:00");
        med2.setElderUser(testElderUser);

        when(medicationReminderRepository.findAll())
                .thenReturn(Arrays.asList(med1, med2));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getAllMedicationsByUser(1L);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals(2, response.getData().size());
        verify(medicationReminderRepository, times(1)).findAll();
    }

    /**
     * Test 10: getAllMedicationsByUser - Empty List
     * Scenario: User has no medications
     * Expected: Return empty list with success
     */
    @Test
    void testGetAllMedicationsByUser_EmptyList() {
        // Arrange
        when(medicationReminderRepository.findAll())
                .thenReturn(Arrays.asList());

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getAllMedicationsByUser(999L);

        // Assert
        assertEquals("success", response.getStatus());
        assertTrue(response.getData().isEmpty());
    }

    // ==================== getStandaloneMedicationsByUser Tests ====================

    /**
     * Test 11: getStandaloneMedicationsByUser - Success
     * Scenario: Get medications without prescription
     * Expected: Return list of standalone medications
     */
    @Test
    void testGetStandaloneMedicationsByUser_Success() {
        // Arrange
        MedicationReminder med1 = new MedicationReminder();
        med1.setId(1L);
        med1.setName("Vitamin C");
        med1.setPrescription(null);
        med1.setType(ETypeMedication.OVER_THE_COUNTER);
        med1.setDescription("");
        med1.setDaysOfWeek("1111111");
        med1.setIsActive(true);
        med1.setReminderTime("08:00");
        med1.setElderUser(testElderUser);

        when(medicationReminderRepository.findAll())
                .thenReturn(Arrays.asList(med1));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getStandaloneMedicationsByUser(1L);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals(1, response.getData().size());
    }

    // ==================== updateMedication Tests ====================

    /**
     * Test 12: updateMedication - Elder Updates Own Medication
     * Scenario: Elder user updates their own medication
     * Expected: Medication updated successfully
     */
    @Test
    void testUpdateMedication_ElderUpdateOwn_Success() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setName("Old Med");
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setName("Updated Med");
        updateRequest.setDescription("New description");

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).save(any(MedicationReminder.class));
    }

    /**
     * Test 13: updateMedication - Supervisor Updates With Permission
     * Scenario: Supervisor with permission updates elder's medication
     * Expected: Medication updated successfully
     */
    @Test
    void testUpdateMedication_SupervisorWithPermission_Success() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setName("Med");
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setName("Updated");

        ElderSupervisor elderSupervisor = new ElderSupervisor();
        // Permission checked by findActiveWithUpdatePermission query itself

        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(supervisorUserRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        when(elderSupervisorRepository.findActiveWithUpdatePermission(2L, 1L))
                .thenReturn(Optional.of(elderSupervisor));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
    }

    /**
     * Test 14: updateMedication - Medication Not Found
     * Scenario: Update non-existent medication
     * Expected: Return error response
     */
    @Test
    void testUpdateMedication_NotFound_ReturnsError() {
        // Arrange
        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        when(medicationReminderRepository.findById(999L)).thenReturn(Optional.empty());

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(999L, updateRequest, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("không tồn tại"));
    }

    // ==================== deleteMedication Tests ====================

    /**
     * Test 15: deleteMedication - Elder Deletes Own Medication
     * Scenario: Elder user deletes their own medication
     * Expected: Medication deleted successfully
     */
    @Test
    void testDeleteMedication_ElderDeleteOwn_Success() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setElderUser(testElderUser);

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act
        BaseResponse<String> response = medicationService.deleteMedication(1L, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).delete(medication);
        verify(simpleTimeBasedScheduler, times(1)).scheduleUserReminders(testElderUser.getId());
    }

    /**
     * Test 16: deleteMedication - Supervisor With Permission
     * Scenario: Supervisor deletes elder's medication with permission
     * Expected: Medication deleted successfully
     */
    @Test
    void testDeleteMedication_SupervisorWithPermission_Success() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setElderUser(testElderUser);

        ElderSupervisor elderSupervisor = new ElderSupervisor();

        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(supervisorUserRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        when(elderSupervisorRepository.findActiveWithUpdatePermission(2L, 1L))
                .thenReturn(Optional.of(elderSupervisor));

        // Act
        BaseResponse<String> response = medicationService.deleteMedication(1L, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).delete(medication);
    }

    /**
     * Test 17: deleteMedication - No Permission
     * Scenario: Supervisor without permission tries to delete
     * Expected: Return error response
     */
    @Test
    void testDeleteMedication_NoPermission_ReturnsError() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setElderUser(testElderUser);

        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(supervisorUserRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        when(elderSupervisorRepository.findActiveWithUpdatePermission(2L, 1L))
                .thenReturn(Optional.empty());

        // Act
        BaseResponse<String> response = medicationService.deleteMedication(1L, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Không có quyền"));
        verify(medicationReminderRepository, never()).delete(any());
    }

    // ==================== toggleMedicationStatus Tests ====================

    /**
     * Test 18: toggleMedicationStatus - Enable Medication
     * Scenario: Toggle inactive medication to active
     * Expected: Status changed to active
     */
    @Test
    void testToggleMedicationStatus_EnableMedication_Success() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setIsActive(false);
        medication.setElderUser(testElderUser);

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.toggleMedicationStatus(1L, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).save(any(MedicationReminder.class));
        verify(simpleTimeBasedScheduler, times(1)).scheduleUserReminders(testElderUser.getId());
    }

    /**
     * Test 19: toggleMedicationStatus - Disable Medication
     * Scenario: Toggle active medication to inactive
     * Expected: Status changed to inactive
     */
    @Test
    void testToggleMedicationStatus_DisableMedication_Success() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setIsActive(true);
        medication.setElderUser(testElderUser);

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.toggleMedicationStatus(1L, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).save(any(MedicationReminder.class));
    }

    /**
     * Test 20: toggleMedicationStatus - Medication Not Found
     * Scenario: Toggle status of non-existent medication
     * Expected: Return error response
     */
    @Test
    void testToggleMedicationStatus_NotFound_ReturnsError() {
        // Arrange
        when(medicationReminderRepository.findById(999L)).thenReturn(Optional.empty());

        // Act
        BaseResponse<MedicationResponse> response = medicationService.toggleMedicationStatus(999L, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("không tồn tại"));
    }

    /**
     * Test 21: toggleMedicationStatus - Supervisor No Permission
     * Scenario: Supervisor without permission tries to toggle
     * Expected: Return error response
     */
    @Test
    void testToggleMedicationStatus_SupervisorNoPermission_ReturnsError() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setIsActive(true);
        medication.setElderUser(testElderUser);

        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(supervisorUserRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        when(elderSupervisorRepository.findActiveWithUpdatePermission(2L, 1L))
                .thenReturn(Optional.empty());

        // Act
        BaseResponse<MedicationResponse> response = medicationService.toggleMedicationStatus(1L, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Không có quyền"));
        verify(medicationReminderRepository, never()).save(any());
    }

    /**
     * Test 22: toggleMedicationStatus - Scheduler Fails But Toggle Succeeds
     * Scenario: Status toggled but rescheduling throws exception
     * Expected: Toggle still succeeds, exception handled gracefully
     * Covers: lines 505-507 - Status toggled but rescheduling failed
     */
    @Test
    void testToggleMedicationStatus_SchedulerFails_ToggleStillSucceeds() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setIsActive(true);
        medication.setElderUser(testElderUser);

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);
        doThrow(new RuntimeException("Scheduler connection timeout"))
                .when(simpleTimeBasedScheduler).scheduleUserReminders(anyLong());

        // Act
        BaseResponse<MedicationResponse> response = medicationService.toggleMedicationStatus(1L, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
        verify(medicationReminderRepository, times(1)).save(any(MedicationReminder.class));
        verify(simpleTimeBasedScheduler, times(1)).scheduleUserReminders(testElderUser.getId());
    }

    // ==================== Edge Cases & Error Handling ====================

    /**
     * Test 22: createMedication - With PrescriptionId
     * Scenario: Create medication linked to a prescription
     * Expected: Medication created with prescription reference
     */
    @Test
    void testCreateMedication_WithPrescription_Success() {
        // Arrange
        createRequest.setElderUserId(null);
        createRequest.setPrescriptionId(100L);
        
        Prescriptions prescription = new Prescriptions();
        prescription.setId(100L);

        when(authentication.getName()).thenReturn("elder@test.com");
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(prescriptionRepository.findById(100L)).thenReturn(Optional.of(prescription));
        when(medicationReminderRepository.save(any(MedicationReminder.class)))
                .thenAnswer(invocation -> {
                    MedicationReminder med = invocation.getArgument(0);
                    med.setId(System.currentTimeMillis());
                    return med;
                });

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(createRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(prescriptionRepository, times(1)).findById(100L);
        verify(medicationReminderRepository, times(2)).save(any(MedicationReminder.class));
    }

    /**
     * Test 23: createMedication - Invalid PrescriptionId
     * Scenario: Create medication with non-existent prescription
     * Expected: Return error response
     */
    @Test
    void testCreateMedication_InvalidPrescription_ReturnsError() {
        // Arrange
        createRequest.setElderUserId(null);
        createRequest.setPrescriptionId(999L);

        when(authentication.getName()).thenReturn("elder@test.com");
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(prescriptionRepository.findById(999L)).thenReturn(Optional.empty());

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(createRequest, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Prescription không tồn tại"));
        verify(medicationReminderRepository, never()).save(any());
    }

    /**
     * Test 24: getMedicationById - Elder Access Different User's Medication
     * Scenario: Elder tries to access another elder's medication
     * Expected: Return error response (no permission)
     */
    @Test
    void testGetMedicationById_DifferentElderUser_ReturnsError() {
        // Arrange
        ElderUser otherElderUser = new ElderUser();
        otherElderUser.setId(99L);
        otherElderUser.setEmail("other@test.com");

        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setElderUser(otherElderUser);

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act
        BaseResponse<MedicationResponse> response = medicationService.getMedicationById(1L, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Không có quyền"));
    }

    /**
     * Test 25: getAllMedicationsByUser - Exception Handling
     * Scenario: Repository throws exception
     * Expected: Return error response gracefully
     */
    @Test
    void testGetAllMedicationsByUser_ExceptionThrown_ReturnsError() {
        // Arrange
        when(medicationReminderRepository.findAll())
                .thenThrow(new RuntimeException("Database connection failed"));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getAllMedicationsByUser(1L);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Lỗi khi lấy danh sách medications"));
    }

    /**
     * Test 26: getStandaloneMedicationsByUser - With OVER_THE_COUNTER Type
     * Scenario: Get medications with OVER_THE_COUNTER type
     * Expected: Return medications including OTC type
     */
    @Test
    void testGetStandaloneMedicationsByUser_OverTheCounter_Success() {
        // Arrange
        MedicationReminder med1 = new MedicationReminder();
        med1.setId(1L);
        med1.setName("Vitamin D");
        med1.setPrescription(null);
        med1.setType(ETypeMedication.OVER_THE_COUNTER);
        med1.setDescription("Daily supplement");
        med1.setDaysOfWeek("1111111");
        med1.setIsActive(true);
        med1.setReminderTime("08:00");
        med1.setElderUser(testElderUser);

        when(medicationReminderRepository.findAll()).thenReturn(Arrays.asList(med1));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getStandaloneMedicationsByUser(1L);

        // Assert
        assertEquals("success", response.getStatus());
        assertFalse(response.getData().isEmpty());
    }

    /**
     * Test 27: getStandaloneMedicationsByUser - Grouped Medications
     * Scenario: Multiple medications with same name should be grouped
     * Expected: Return grouped medications with combined reminder times
     */
    @Test
    void testGetStandaloneMedicationsByUser_GroupedMedications_Success() {
        // Arrange
        MedicationReminder med1 = new MedicationReminder();
        med1.setId(1L);
        med1.setName("Vitamin C");
        med1.setType(ETypeMedication.OVER_THE_COUNTER);
        med1.setPrescription(null);
        med1.setDescription("Morning dose");
        med1.setDaysOfWeek("1111111");
        med1.setIsActive(true);
        med1.setReminderTime("08:00");
        med1.setElderUser(testElderUser);

        MedicationReminder med2 = new MedicationReminder();
        med2.setId(2L);
        med2.setName("Vitamin C");
        med2.setType(ETypeMedication.OVER_THE_COUNTER);
        med2.setPrescription(null);
        med2.setDescription("Morning dose");
        med2.setDaysOfWeek("1111111");
        med2.setIsActive(true);
        med2.setReminderTime("20:00");
        med2.setElderUser(testElderUser);

        when(medicationReminderRepository.findAll()).thenReturn(Arrays.asList(med1, med2));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getStandaloneMedicationsByUser(1L);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals(1, response.getData().size()); // Should be grouped into 1
        assertEquals(2, response.getData().get(0).getReminderTimes().size()); // With 2 reminder times
    }

    /**
     * Test 28: updateMedication - Update ReminderTimes and Reschedule
     * Scenario: Update reminder times should trigger rescheduling
     * Expected: Medication updated and scheduler called
     */
    @Test
    void testUpdateMedication_UpdateReminderTime_TriggersReschedule() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setName("Med");
        medication.setReminderTime("08:00");
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setReminderTimes(Arrays.asList("12:00", "18:00"));

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(simpleTimeBasedScheduler, times(1)).scheduleUserReminders(testElderUser.getId());
    }

    /**
     * Test 29: updateMedication - Partial Update (Only Description)
     * Scenario: Update only description field, keep others unchanged
     * Expected: Only description updated
     */
    @Test
    void testUpdateMedication_PartialUpdate_Success() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setName("Original Med");
        medication.setDescription("Original description");
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setDescription("Updated description only");
        // Name and other fields are null (not updated)

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).save(any(MedicationReminder.class));
    }

    /**
     * Test 30: deleteMedication - Medication Not Found
     * Scenario: Delete non-existent medication
     * Expected: Return error response
     */
    @Test
    void testDeleteMedication_NotFound_ReturnsError() {
        // Arrange
        when(medicationReminderRepository.findById(999L)).thenReturn(Optional.empty());

        // Act
        BaseResponse<String> response = medicationService.deleteMedication(999L, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("không tồn tại"));
        verify(medicationReminderRepository, never()).delete(any());
    }

    /**
     * Test 31: deleteMedication - Scheduling Fails But Delete Succeeds
     * Scenario: Medication deleted even if rescheduling fails
     * Expected: Return success, medication deleted
     */
    @Test
    void testDeleteMedication_SchedulingFails_StillDeletes() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setElderUser(testElderUser);

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        doThrow(new RuntimeException("Scheduler error")).when(simpleTimeBasedScheduler).scheduleUserReminders(anyLong());

        // Act
        BaseResponse<String> response = medicationService.deleteMedication(1L, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).delete(medication);
    }

    /**
     * Test 32: createMedication - Supervisor User Not Found
     * Scenario: Supervisor email not found in database
     * Expected: Return error response
     */
    @Test
    void testCreateMedication_SupervisorNotFound_ReturnsError() {
        // Arrange
        createRequest.setElderUserId(1L);
        when(authentication.getName()).thenReturn("unknown@test.com");
        when(elderUserRepository.findById(1L)).thenReturn(Optional.of(testElderUser));
        when(supervisorUserRepository.findByEmail("unknown@test.com")).thenReturn(Optional.empty());

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(createRequest, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("User không tồn tại"));
    }

    /**
     * Test 33: createMedication - Elder User Not Found (Self Create)
     * Scenario: Elder tries to create for themselves but email not found
     * Expected: Return error response
     */
    @Test
    void testCreateMedication_ElderNotFoundSelfCreate_ReturnsError() {
        // Arrange
        createRequest.setElderUserId(null);
        when(authentication.getName()).thenReturn("unknown@test.com");
        when(elderUserRepository.findByEmail("unknown@test.com")).thenReturn(Optional.empty());

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.createMedication(createRequest, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Elder user không tồn tại"));
    }

    /**
     * Test 34: updateMedication - Update With Supervisor Permission
     * Scenario: Supervisor with permission updates medication
     * Expected: Medication updated and rescheduled
     */
    @Test
    void testUpdateMedication_SupervisorWithPermission_TriggersReschedule() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setName("Med");
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setName("Updated Med");
        updateRequest.setReminderTimes(Arrays.asList("10:00"));

        ElderSupervisor elderSupervisor = new ElderSupervisor();

        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(supervisorUserRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        when(elderSupervisorRepository.findActiveWithUpdatePermission(2L, 1L))
                .thenReturn(Optional.of(elderSupervisor));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(simpleTimeBasedScheduler, times(1)).scheduleUserReminders(testElderUser.getId());
    }

    /**
     * Test 35: updateMedication - Supervisor No Permission
     * Scenario: Supervisor without permission tries to update
     * Expected: Return error response
     */
    @Test
    void testUpdateMedication_SupervisorNoPermission_ReturnsError() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setName("Updated");

        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(supervisorUserRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        when(elderSupervisorRepository.findActiveWithUpdatePermission(2L, 1L))
                .thenReturn(Optional.empty());

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Không có quyền"));
        verify(medicationReminderRepository, never()).save(any());
    }

    // ==================== Additional Coverage Tests ====================

    /**
     * Test 36: getMedicationById - Exception During Retrieval
     * Scenario: Repository throws exception (covers line 174-179)
     * Expected: Return error response
     */
    @Test
    void testGetMedicationById_ExceptionThrown_ReturnsError() {
        // Arrange
        when(medicationReminderRepository.findById(1L))
                .thenThrow(new RuntimeException("Database error"));

        // Act
        BaseResponse<MedicationResponse> response = medicationService.getMedicationById(1L, authentication);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Lỗi khi lấy medication"));
        assertNull(response.getData());
    }

    /**
     * Test 37: getStandaloneMedicationsByUser - With Prescription (covers line 226-227)
     * Scenario: Medication has prescription but is OVER_THE_COUNTER type
     * Expected: Should be included in standalone list
     */
    @Test
    void testGetStandaloneMedicationsByUser_WithPrescriptionButOTC_Included() {
        // Arrange
        Prescriptions prescription = new Prescriptions();
        prescription.setId(100L);

        MedicationReminder med1 = new MedicationReminder();
        med1.setId(1L);
        med1.setName("OTC Med");
        med1.setPrescription(prescription); // Has prescription
        med1.setType(ETypeMedication.OVER_THE_COUNTER); // But OTC type
        med1.setDescription("Test");
        med1.setDaysOfWeek("1111111");
        med1.setIsActive(true);
        med1.setReminderTime("08:00");
        med1.setElderUser(testElderUser);

        when(medicationReminderRepository.findAll()).thenReturn(Arrays.asList(med1));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getStandaloneMedicationsByUser(1L);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals(1, response.getData().size());
    }

    /**
     * Test 38: getStandaloneMedicationsByUser - With Prescription Info (covers line 261)
     * Scenario: Medication has prescription, verify prescriptionId in response
     * Expected: Response includes prescriptionId
     */
    @Test
    void testGetStandaloneMedicationsByUser_WithPrescriptionId_IncludedInResponse() {
        // Arrange
        Prescriptions prescription = new Prescriptions();
        prescription.setId(123L);

        MedicationReminder med1 = new MedicationReminder();
        med1.setId(1L);
        med1.setName("Med");
        med1.setPrescription(prescription);
        med1.setType(ETypeMedication.OVER_THE_COUNTER);
        med1.setDescription("With prescription");
        med1.setDaysOfWeek("1111111");
        med1.setIsActive(true);
        med1.setReminderTime("08:00");
        med1.setElderUser(testElderUser);

        when(medicationReminderRepository.findAll()).thenReturn(Arrays.asList(med1));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getStandaloneMedicationsByUser(1L);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData().get(0).getPrescriptionId());
        assertEquals(123L, response.getData().get(0).getPrescriptionId());
    }

    /**
     * Test 39: getStandaloneMedicationsByUser - With Timestamps (covers line 268-270)
     * Scenario: Medication has createdAt and updatedAt
     * Expected: Timestamps included in response
     */
    @Test
    void testGetStandaloneMedicationsByUser_WithTimestamps_IncludedInResponse() {
        // Arrange
        MedicationReminder med1 = new MedicationReminder();
        med1.setId(1L);
        med1.setName("Med");
        med1.setPrescription(null);
        med1.setType(ETypeMedication.OVER_THE_COUNTER);
        med1.setDescription("Test");
        med1.setDaysOfWeek("1111111");
        med1.setIsActive(true);
        med1.setReminderTime("08:00");
        med1.setElderUser(testElderUser);
        med1.setCreatedAt(java.time.Instant.now());
        med1.setUpdatedAt(java.time.Instant.now());

        when(medicationReminderRepository.findAll()).thenReturn(Arrays.asList(med1));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getStandaloneMedicationsByUser(1L);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData().get(0).getCreatedAt());
        assertNotNull(response.getData().get(0).getUpdatedAt());
    }

    /**
     * Test 40: getStandaloneMedicationsByUser - Exception Handling (covers line 282-287)
     * Scenario: Repository throws exception during retrieval
     * Expected: Return error response gracefully
     */
    @Test
    void testGetStandaloneMedicationsByUser_ExceptionThrown_ReturnsError() {
        // Arrange
        when(medicationReminderRepository.findAll())
                .thenThrow(new RuntimeException("Database connection lost"));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getStandaloneMedicationsByUser(1L);

        // Assert
        assertEquals("error", response.getStatus());
        assertTrue(response.getMessage().contains("Lỗi khi lấy danh sách standalone medications"));
        assertNull(response.getData());
    }

    /**
     * Test 41: updateMedication - Update Type Field (covers line 337-339)
     * Scenario: Update only the type field
     * Expected: Type updated successfully
     */
    @Test
    void testUpdateMedication_UpdateTypeField_Success() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setName("Med");
        medication.setType(ETypeMedication.OVER_THE_COUNTER);
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setType(ETypeMedication.OVER_THE_COUNTER);

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).save(any(MedicationReminder.class));
    }

    /**
     * Test 42: updateMedication - Update DaysOfWeek Field (covers line 340-342)
     * Scenario: Update only the daysOfWeek field
     * Expected: DaysOfWeek updated successfully
     */
    @Test
    void testUpdateMedication_UpdateDaysOfWeekField_Success() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setName("Med");
        medication.setDaysOfWeek("1111111");
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setDaysOfWeek("1010101"); // Weekdays only

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).save(any(MedicationReminder.class));
    }

    /**
     * Test 43: updateMedication - Update IsActive Field (covers line 343-345)
     * Scenario: Update only the isActive field
     * Expected: IsActive updated successfully
     */
    @Test
    void testUpdateMedication_UpdateIsActiveField_Success() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setName("Med");
        medication.setIsActive(true);
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setIsActive(false);

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).save(any(MedicationReminder.class));
    }

    /**
     * Test 44: updateMedication - Update With Empty ReminderTimes (covers line 348)
     * Scenario: Update request has empty reminderTimes list
     * Expected: ReminderTime not updated
     */
    @Test
    void testUpdateMedication_EmptyReminderTimes_NotUpdated() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setName("Med");
        medication.setReminderTime("08:00");
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setName("Updated Med");
        updateRequest.setReminderTimes(Arrays.asList()); // Empty list

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(medicationReminderRepository, times(1)).save(any(MedicationReminder.class));
    }

    /**
     * Test 45: updateMedication - Scheduler Exception Handling (covers line 365-366)
     * Scenario: Update succeeds but scheduler throws exception
     * Expected: Update still succeeds, error logged
     */
    @Test
    void testUpdateMedication_SchedulerFails_UpdateStillSucceeds() {
        // Arrange
        MedicationReminder medication = new MedicationReminder();
        medication.setId(1L);
        medication.setName("Med");
        medication.setElderUser(testElderUser);

        UpdateMedicationRequest updateRequest = new UpdateMedicationRequest();
        updateRequest.setName("Updated Med");

        when(authentication.getName()).thenReturn("elder@test.com");
        when(medicationReminderRepository.findById(1L)).thenReturn(Optional.of(medication));
        when(elderUserRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(medicationReminderRepository.save(any(MedicationReminder.class))).thenReturn(medication);
        doThrow(new RuntimeException("Scheduler error")).when(simpleTimeBasedScheduler).scheduleUserReminders(anyLong());

        // Act
        BaseResponse<MedicationResponse> response = medicationService.updateMedication(1L, updateRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
        verify(medicationReminderRepository, times(1)).save(any(MedicationReminder.class));
        verify(simpleTimeBasedScheduler, times(1)).scheduleUserReminders(testElderUser.getId());
    }

    /**
     * Test 46: getStandaloneMedicationsByUser - Null Timestamps (covers line 268-270 null case)
     * Scenario: Medication has null createdAt and updatedAt
     * Expected: Response handles null timestamps gracefully
     */
    @Test
    void testGetStandaloneMedicationsByUser_NullTimestamps_HandledGracefully() {
        // Arrange
        MedicationReminder med1 = new MedicationReminder();
        med1.setId(1L);
        med1.setName("Med");
        med1.setPrescription(null);
        med1.setType(ETypeMedication.OVER_THE_COUNTER);
        med1.setDescription("Test");
        med1.setDaysOfWeek("1111111");
        med1.setIsActive(true);
        med1.setReminderTime("08:00");
        med1.setElderUser(testElderUser);
        med1.setCreatedAt(null);
        med1.setUpdatedAt(null);

        when(medicationReminderRepository.findAll()).thenReturn(Arrays.asList(med1));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getStandaloneMedicationsByUser(1L);

        // Assert
        assertEquals("success", response.getStatus());
        assertNull(response.getData().get(0).getCreatedAt());
        assertNull(response.getData().get(0).getUpdatedAt());
    }

    /**
     * Test 47: getStandaloneMedicationsByUser - Different User IDs Filtered (covers line 225)
     * Scenario: Repository has medications from multiple users
     * Expected: Only medications for specified user returned
     */
    @Test
    void testGetStandaloneMedicationsByUser_FiltersByUserId_Success() {
        // Arrange
        ElderUser otherUser = new ElderUser();
        otherUser.setId(99L);
        otherUser.setEmail("other@test.com");

        MedicationReminder med1 = new MedicationReminder();
        med1.setId(1L);
        med1.setName("Med1");
        med1.setPrescription(null);
        med1.setType(ETypeMedication.OVER_THE_COUNTER);
        med1.setDescription("Test");
        med1.setDaysOfWeek("1111111");
        med1.setIsActive(true);
        med1.setReminderTime("08:00");
        med1.setElderUser(testElderUser); // User ID 1

        MedicationReminder med2 = new MedicationReminder();
        med2.setId(2L);
        med2.setName("Med2");
        med2.setPrescription(null);
        med2.setType(ETypeMedication.OVER_THE_COUNTER);
        med2.setDescription("Test");
        med2.setDaysOfWeek("1111111");
        med2.setIsActive(true);
        med2.setReminderTime("08:00");
        med2.setElderUser(otherUser); // User ID 99

        when(medicationReminderRepository.findAll()).thenReturn(Arrays.asList(med1, med2));

        // Act
        BaseResponse<List<MedicationResponse>> response = medicationService.getStandaloneMedicationsByUser(1L);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals(1, response.getData().size());
        assertEquals(1L, response.getData().get(0).getUserId());
    }
}

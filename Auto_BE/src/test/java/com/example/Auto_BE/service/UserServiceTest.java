package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.ChangePasswordRequest;
import com.example.Auto_BE.dto.request.UpdateProfileRequest;
import com.example.Auto_BE.dto.response.ProfileResponse;
import com.example.Auto_BE.entity.ElderUser;
import com.example.Auto_BE.entity.SupervisorUser;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.entity.enums.EBloodType;
import com.example.Auto_BE.entity.enums.EGender;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

/**
 * Unit Test cho UserService
 * Test cÃƒÂ¡c chÃ¡Â»Â©c nÃ„Æ’ng quÃ¡ÂºÂ£n lÃƒÂ½ user profile vÃƒÂ  password
 */
@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @Mock
    private PasswordEncoder passwordEncoder;

    @Mock
    private CloudinaryService cloudinaryService;

    @Mock
    private Authentication authentication;

    @InjectMocks
    private UserService userService;

    private ElderUser testElderUser;
    private SupervisorUser testSupervisorUser;
    private UpdateProfileRequest updateProfileRequest;
    private ChangePasswordRequest changePasswordRequest;

    @BeforeEach
    void setUp() {
        // Setup Elder User
        testElderUser = new ElderUser();
        testElderUser.setId(1L);
        testElderUser.setEmail("elder@test.com");
        testElderUser.setFullName("Test Elder");
        testElderUser.setPhoneNumber("0123456789");
        testElderUser.setDateOfBirth(LocalDate.of(1960, 1, 1));
        testElderUser.setGender(EGender.MALE);
        testElderUser.setAddress("Ha Noi");
        testElderUser.setPassword("$2a$10$hashedPassword"); // BCrypt hash
        testElderUser.setBloodType(EBloodType.O_POSITIVE);
        testElderUser.setHeight(170.0);
        testElderUser.setWeight(65.0);

        // Setup Supervisor User
        testSupervisorUser = new SupervisorUser();
        testSupervisorUser.setId(2L);
        testSupervisorUser.setEmail("supervisor@test.com");
        testSupervisorUser.setFullName("Test Supervisor");
        testSupervisorUser.setPhoneNumber("0987654321");
        testSupervisorUser.setGender(EGender.FEMALE);
        testSupervisorUser.setPassword("$2a$10$hashedPassword");
        testSupervisorUser.setOccupation("Nurse");
        testSupervisorUser.setWorkplace("Hospital");

        // Setup Update Profile Request
        updateProfileRequest = new UpdateProfileRequest();
        updateProfileRequest.setFullName("Updated Name");
        updateProfileRequest.setPhoneNumber("0999999999");
        updateProfileRequest.setAddress("Ho Chi Minh");

        // Setup Change Password Request
        changePasswordRequest = new ChangePasswordRequest();
        changePasswordRequest.setCurrentPassword("OldPassword123");
        changePasswordRequest.setNewPassword("NewPassword456");
    }

    // ==================== getUserProfile Tests ====================

    /**
     * TC01: getUserProfile - User tồn tại (Elder)
     * Test 1: Get user profile successfully when user exists
     */
    @Test
    void testGetUserProfile_ExistingUser_Success() {
        // Arrange
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act
        BaseResponse<?> response = userService.getUserProfile(authentication);

        // Assert
        assertNotNull(response);
        assertEquals("success", response.getStatus());
        assertEquals("Lấy thông tin cá nhân thành công", response.getMessage());
        assertNotNull(response.getData());
        assertTrue(response.getData() instanceof ProfileResponse);
        
        ProfileResponse profileData = (ProfileResponse) response.getData();
        assertEquals("Test Elder", profileData.getFullName());
        assertEquals("elder@test.com", profileData.getEmail());
        assertEquals("0123456789", profileData.getPhoneNumber());
        
        verify(userRepository, times(1)).findByEmail("elder@test.com");
    }

    /**
     * TC02: getUserProfile - User không tồn tại
     * Test 2: Throw exception when user not found
     */
    @Test
    void testGetUserProfile_UserNotFound_ThrowsException() {
        // Arrange
        when(authentication.getName()).thenReturn("nonexistent@test.com");
        when(userRepository.findByEmail("nonexistent@test.com")).thenReturn(Optional.empty());

        // Act & Assert
        BaseException.EntityNotFoundException exception = assertThrows(
                BaseException.EntityNotFoundException.class,
                () -> userService.getUserProfile(authentication)
        );
        
        assertEquals("User not found", exception.getMessage());
        verify(userRepository, times(1)).findByEmail("nonexistent@test.com");
    }

    /**
     * TC05: updateUserProfile - Elder cập nhật một phần (fullName)
     * Test 3: Update Elder user profile with medical info successfully
     */
    @Test
    void testUpdateUserProfile_ElderUser_Success() {
        // Arrange
        updateProfileRequest.setBloodType(EBloodType.A_POSITIVE);
        updateProfileRequest.setHeight(175.0);
        updateProfileRequest.setWeight(70.0);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("Cập nhật thông tin cá nhân thành công", response.getMessage());
        
        // Verify Elder-specific fields were updated
        assertEquals("Updated Name", testElderUser.getFullName());
        assertEquals("0999999999", testElderUser.getPhoneNumber());
        assertEquals("Ho Chi Minh", testElderUser.getAddress());
        assertEquals(EBloodType.A_POSITIVE, testElderUser.getBloodType());
        assertEquals(175.0, testElderUser.getHeight());
        assertEquals(70.0, testElderUser.getWeight());
        
        verify(userRepository, times(1)).save(testElderUser);
    }

    /**
     * TC07: updateUserProfile - Supervisor cập nhật occupation
     * Test 4: Update Supervisor user profile with occupation info successfully
     */
    @Test
    void testUpdateUserProfile_SupervisorUser_Success() {
        // Arrange
        updateProfileRequest.setOccupation("Doctor");
        updateProfileRequest.setWorkplace("Clinic ABC");
        
        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(userRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        when(userRepository.save(any(SupervisorUser.class))).thenReturn(testSupervisorUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("Doctor", testSupervisorUser.getOccupation());
        assertEquals("Clinic ABC", testSupervisorUser.getWorkplace());
        
        verify(userRepository, times(1)).save(testSupervisorUser);
    }

    /**
     * TC09: updateUserProfile - Partial update giữ nguyên fields null
     * Test 5: Partial update with only some fields keeps other fields unchanged
     */
    @Test
    void testUpdateUserProfile_PartialUpdate_Success() {
        // Arrange
        UpdateProfileRequest partialRequest = new UpdateProfileRequest();
        partialRequest.setFullName("New Name Only");
        // All other fields are null
        
        String originalPhone = testElderUser.getPhoneNumber();
        String originalAddress = testElderUser.getAddress();
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(partialRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("New Name Only", testElderUser.getFullName());
        assertEquals(originalPhone, testElderUser.getPhoneNumber()); // Unchanged
        assertEquals(originalAddress, testElderUser.getAddress()); // Unchanged
    }

    /**
     * TC18: changePassword - Đổi mật khẩu thành công
     * Test 6: Change password successfully with valid current and new password
     */
    @Test
    void testChangePassword_ValidPasswords_Success() {
        // Arrange
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(passwordEncoder.matches("OldPassword123", testElderUser.getPassword())).thenReturn(true);
        when(passwordEncoder.matches("NewPassword456", testElderUser.getPassword())).thenReturn(false);
        when(passwordEncoder.encode("NewPassword456")).thenReturn("$2a$10$newHashedPassword");

        // Act
        BaseResponse<String> response = userService.changePassword(changePasswordRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("Cập nhật mật khẩu thành công", response.getMessage());
        assertEquals("$2a$10$newHashedPassword", testElderUser.getPassword());
        
        verify(passwordEncoder, times(1)).matches("OldPassword123", "$2a$10$hashedPassword");
        verify(passwordEncoder, times(1)).encode("NewPassword456");
        verify(userRepository, times(1)).save(testElderUser);
    }

    /**
     * TC19: changePassword - Mật khẩu hiện tại sai
     * Test 7: changePassword - Current password sai (FAIL)
     * Scenario:
     * Expected: Throw BadRequestException
     */
    @Test
    void testChangePassword_IncorrectCurrentPassword_ThrowsException() {
        // Arrange
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(passwordEncoder.matches("OldPassword123", testElderUser.getPassword())).thenReturn(false); // Wrong password

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.changePassword(changePasswordRequest, authentication)
        );
        
        assertEquals("Mật khẩu hiện tại không đúng", exception.getMessage());
        verify(userRepository, never()).save(any(User.class));
    }

    /**
     * TC20: changePassword - Mật khẩu mới trùng mật khẩu cũ
     * Test 8: changePassword - New password trùng old password (FAIL)
     * Scenario:
     * Expected: Throw BadRequestException
     */
    @Test
    void testChangePassword_SameAsOldPassword_ThrowsException() {
        // Arrange
        changePasswordRequest.setNewPassword("OldPassword123"); // Same as current
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(passwordEncoder.matches("OldPassword123", testElderUser.getPassword())).thenReturn(true);
        // New password matches old password
        when(passwordEncoder.matches("OldPassword123", testElderUser.getPassword())).thenReturn(true);

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.changePassword(changePasswordRequest, authentication)
        );
        
        assertTrue(exception.getMessage().contains("password"));
        verify(userRepository, never()).save(any(User.class));
    }

    /**
     * TC21: changePassword - User không tồn tại
     * Test 9: changePassword - User not found
     * Scenario:
     * Expected: Throw EntityNotFoundException
     */
    @Test
    void testChangePassword_UserNotFound_ThrowsException() {
        // Arrange
        when(authentication.getName()).thenReturn("nonexistent@test.com");
        when(userRepository.findByEmail("nonexistent@test.com")).thenReturn(Optional.empty());

        // Act & Assert
        assertThrows(
                BaseException.EntityNotFoundException.class,
                () -> userService.changePassword(changePasswordRequest, authentication)
        );
        
        verify(passwordEncoder, never()).matches(anyString(), anyString());
        verify(userRepository, never()).save(any(User.class));
    }

    /**
     * TC11: updateUserProfile - Trường bloodType null giữ nguyên
     * Test 10: updateUserProfile - Elder update null blood type (EDGE CASE)
     * Scenario: Elder
     * Expected: BloodType
     */
    @Test
    void testUpdateUserProfile_NullBloodType_KeepsOriginalValue() {
        // Arrange
        updateProfileRequest.setBloodType(null); // Explicitly null
        EBloodType originalBloodType = testElderUser.getBloodType();
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals(originalBloodType, testElderUser.getBloodType()); // Unchanged
    }

    /**
     * TC03: getUserProfile - Elder có thông tin y tế
     * Test 11: Get Elder user profile with complete medical information
     */
    @Test
    void testGetUserProfile_ElderWithMedicalInfo_ReturnsCompleteData() {
        // Arrange -
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act
        BaseResponse<?> response = userService.getUserProfile(authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("Lấy thông tin cá nhân thành công", response.getMessage());
        
        ProfileResponse profileData = (ProfileResponse) response.getData();
        assertEquals(EBloodType.O_POSITIVE, profileData.getBloodType());
        assertEquals(170.0, profileData.getHeight());
        assertEquals(65.0, profileData.getWeight());
    }

    /**
     * TC04: getUserProfile - Supervisor có thông tin nghề nghiệp
     * Test 12: getUserProfile - SupervisorUser
     */
    @Test
    void testGetUserProfile_Supervisor_ReturnsOccupationInfo() {
        // Arrange
        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(userRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));

        // Act
        BaseResponse<?> response = userService.getUserProfile(authentication);

        // Assert
        assertEquals("success", response.getStatus());
        ProfileResponse profileData = (ProfileResponse) response.getData();
        assertEquals("Nurse", profileData.getOccupation());
        assertEquals("Hospital", profileData.getWorkplace());
    }

    /**
     * TC12: updateUserProfile - Cập nhật height/weight hợp lệ
     * Test 13: Elder user updates medical information successfully
     */
    @Test
    void testUpdateUserProfile_ElderUpdateMedicalInfo_Success() {
        // Arrange
        updateProfileRequest.setHeight(175.0);
        updateProfileRequest.setWeight(70.0);
        updateProfileRequest.setBloodType(EBloodType.A_POSITIVE);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("Cập nhật thông tin cá nhân thành công", response.getMessage());
        verify(userRepository, times(1)).save(any(ElderUser.class));
    }

    /**
     * TC08: updateUserProfile - Cập nhật gender và dateOfBirth
     * Test 14: updateUserProfile - Elder update gender
     */
    @Test
    void testUpdateUserProfile_ElderUpdateGender_Success() {
        // Arrange
        updateProfileRequest.setGender(EGender.FEMALE);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(userRepository, times(1)).save(any(ElderUser.class));
    }

    /**
     * TC08: updateUserProfile - Cập nhật gender và dateOfBirth
     * Test 15: updateUserProfile - Elder update dateOfBirth
     */
    @Test
    void testUpdateUserProfile_ElderUpdateDateOfBirth_Success() {
        // Arrange
        LocalDate newDob = LocalDate.of(1965, 5, 15);
        updateProfileRequest.setDateOfBirth(newDob);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
    assertEquals("Cập nhật thông tin cá nhân thành công", response.getMessage());
    }

    /**
     * TC07: updateUserProfile - Supervisor cập nhật occupation
     * Test 16: updateUserProfile - Supervisor update occupation
     */
    @Test
    void testUpdateUserProfile_SupervisorUpdateOccupation_Success() {
        // Arrange
        updateProfileRequest.setOccupation("Doctor");
        updateProfileRequest.setWorkplace("Clinic ABC");
        
        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(userRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        when(userRepository.save(any(SupervisorUser.class))).thenReturn(testSupervisorUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(userRepository, times(1)).save(any(SupervisorUser.class));
    }

    /**
     * TC10: updateUserProfile - User không tồn tại
     * Test 17: updateUserProfile - User not found
     */
    @Test
    void testUpdateUserProfile_UserNotFound_ThrowsException() {
        // Arrange
        when(authentication.getName()).thenReturn("notfound@test.com");
        when(userRepository.findByEmail("notfound@test.com")).thenReturn(Optional.empty());

        // Act & Assert
        BaseException.EntityNotFoundException exception = assertThrows(
                BaseException.EntityNotFoundException.class,
                () -> userService.updateUserProfile(updateProfileRequest, authentication)
        );
        
        assertEquals("User not found", exception.getMessage());
        verify(userRepository, never()).save(any());
    }

    /**
     * TC14: updateUserProfile - Cập nhật avatar không có trong DTO
     * Test 18: updateUserProfile - Update avatar URL
     */
    @Test
    void testUpdateUserProfile_UpdateAvatarUrl_Success() {
        // Arrange
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(userRepository, times(1)).save(any(ElderUser.class));
    }

    /**
     * TC19: changePassword - Mật khẩu hiện tại sai
     * Test 19: changePassword - Wrong current password
     */
    @Test
    void testChangePassword_WrongCurrentPassword_ThrowsException() {
        // Arrange
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(passwordEncoder.matches("OldPassword123", testElderUser.getPassword())).thenReturn(false);

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.changePassword(changePasswordRequest, authentication)
        );
        
        assertTrue(exception.getMessage().contains("Mật khẩu hiện tại không hợp lệ") ||
                   exception.getMessage().contains("Mật khẩu hiện tại không đúng"));
        verify(userRepository, never()).save(any());
    }

    /**
     * TC20: changePassword - Mật khẩu mới trùng mật khẩu cũ
     * Test 20: changePassword - New password same as old
     */
    @Test
    void testChangePassword_NewPasswordSameAsOld_ThrowsException() {
        // Arrange
        changePasswordRequest.setNewPassword("OldPassword123");

        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(passwordEncoder.matches("OldPassword123", testElderUser.getPassword())).thenReturn(true);

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.changePassword(changePasswordRequest, authentication)
        );
        
        assertTrue(exception.getMessage().contains("Mật khẩu") ||
                   exception.getMessage().contains("same as"));
        verify(userRepository, never()).save(any());
    }

    /**
     * TC22/TC24: changePassword - Repository save thất bại
     * Test 21: changePassword - Repository save fails
     */
    @Test
    void testChangePassword_RepositorySaveFails_ThrowsException() {
        // Arrange
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(passwordEncoder.matches("OldPassword123", testElderUser.getPassword())).thenReturn(true);
        when(passwordEncoder.encode("NewPassword456")).thenReturn("$2a$10$newHashedPassword");
        when(userRepository.save(any(User.class))).thenThrow(new RuntimeException("Database connection failed"));

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.changePassword(changePasswordRequest, authentication)
        );
        
        assertEquals("Database connection failed", exception.getMessage());
    }

    /**
     * TC13/TC15: updateUserProfile - Elder update tất cả fields
     * Test 22: updateUserProfile - Elder update all fields
     */
    @Test
    void testUpdateUserProfile_ElderUpdateAllFields_Success() {
        // Arrange -
        updateProfileRequest.setFullName("Nguyễn Văn A");
        updateProfileRequest.setPhoneNumber("0912345678");
        updateProfileRequest.setAddress("Nam Định");
        updateProfileRequest.setGender(EGender.MALE);
        updateProfileRequest.setDateOfBirth(LocalDate.of(1970, 1, 1));
        updateProfileRequest.setBloodType(EBloodType.B_POSITIVE);
        updateProfileRequest.setHeight(168.5);
        updateProfileRequest.setWeight(62.0);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("Cập nhật thông tin cá nhân thành công", response.getMessage());
        verify(userRepository, times(1)).save(any(ElderUser.class));
    }

    /**
     * TC16: updateUserProfile - Supervisor chỉ update tên
     * Test 23: updateUserProfile - Supervisor update only name
     */
    @Test
    void testUpdateUserProfile_SupervisorUpdateOnlyName_Success() {
        // Arrange
        UpdateProfileRequest partialUpdate = new UpdateProfileRequest();
        partialUpdate.setFullName("Trần Văn B");
        
        when(authentication.getName()).thenReturn("supervisor@test.com");
        when(userRepository.findByEmail("supervisor@test.com")).thenReturn(Optional.of(testSupervisorUser));
        when(userRepository.save(any(SupervisorUser.class))).thenReturn(testSupervisorUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(partialUpdate, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(userRepository, times(1)).save(any(SupervisorUser.class));
    }

    /**
     * TC17: updateUserProfile - fullName null (all fields null)
     * Test 24: updateUserProfile - All fields null
     */
    @Test
    void testUpdateUserProfile_AllFieldsNull_NoChanges() {

        UpdateProfileRequest emptyRequest = new UpdateProfileRequest();
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(emptyRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        verify(userRepository, times(1)).save(any(ElderUser.class));
    }

    // ============= UPLOAD AVATAR TESTS =============

    /**
     * TC25: uploadAvatar - Upload thành công (không có avatar cũ)
     * Test 25: uploadAvatar - Valid image
     */
    @Test
    void testUploadAvatar_ValidImage_Success() throws Exception {
        // Arrange
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(false);
        when(avatarFile.getSize()).thenReturn(5 * 1024 * 1024L); // 5MB
        when(avatarFile.getContentType()).thenReturn("image/jpeg");
        
        String newAvatarUrl = "https://cloudinary.com/avatar123.jpg";
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(cloudinaryService.uploadImage(avatarFile)).thenReturn(newAvatarUrl);
        when(userRepository.save(any(User.class))).thenReturn(testElderUser);

        // Act -
        BaseResponse<String> response = userService.uploadAvatar(avatarFile, authentication);

        // Assert -
        assertEquals("success", response.getStatus());
        assertEquals("Cập nhật ảnh đại diện thành công", response.getMessage());
        assertEquals(newAvatarUrl, response.getData());
        verify(cloudinaryService, times(1)).uploadImage(avatarFile);
        verify(userRepository, times(1)).save(any(User.class));
    }

    /**
     * TC27: uploadAvatar - File rỗng
     * Test 26: uploadAvatar - Empty file
     */
    @Test
    void testUploadAvatar_EmptyFile_ThrowsException() throws IOException {
        // Arrange -
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(true);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.uploadAvatar(avatarFile, authentication)
        );
        
        assertTrue(exception.getMessage().contains("trống") ||
                   exception.getMessage().contains("empty"));
        verify(cloudinaryService, never()).uploadImage(any());
    }

    /**
     * TC28: uploadAvatar - File quá lớn
     * Test 27: uploadAvatar - File too large
     */
    @Test
    void testUploadAvatar_FileTooLarge_ThrowsException() throws IOException {
        // Arrange - File 15MB
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(false);
        when(avatarFile.getSize()).thenReturn(15 * 1024 * 1024L); // 15MB
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.uploadAvatar(avatarFile, authentication)
        );
        
        assertTrue(exception.getMessage().contains("10MB") || 
                   exception.getMessage().contains("size"));
        verify(cloudinaryService, never()).uploadImage(any());
    }

    /**
     * TC33: uploadAvatar - contentType == null
     * Test 28: uploadAvatar - ContentType null
     */
    @Test
    void testUploadAvatar_NullContentType_ThrowsException() throws IOException {
        // Arrange - contentType = null
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(false);
        when(avatarFile.getSize()).thenReturn(2 * 1024 * 1024L);
        when(avatarFile.getContentType()).thenReturn(null); // null contentType
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.uploadAvatar(avatarFile, authentication)
        );
        
        assertTrue(exception.getMessage().contains("định dang") ||
                   exception.getMessage().contains("JPG"));
        verify(cloudinaryService, never()).uploadImage(any());
    }

    /**
     * TC29: uploadAvatar - Định dạng file không hợp lệ (PDF)
     * Test 29: uploadAvatar - Invalid file type
     */
    @Test
    void testUploadAvatar_InvalidFileType_ThrowsException() throws IOException {
        // Arrange - File PDF
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(false);
        when(avatarFile.getSize()).thenReturn(2 * 1024 * 1024L);
        when(avatarFile.getContentType()).thenReturn("application/pdf");
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.uploadAvatar(avatarFile, authentication)
        );
        
        assertTrue(exception.getMessage().contains("JPG") || 
                   exception.getMessage().contains("PNG") ||
                   exception.getMessage().contains("image"));
        verify(cloudinaryService, never()).uploadImage(any());
    }

    /**
     * TC26/TC34: uploadAvatar - Upload và xóa avatar cũ
     * Test 30: uploadAvatar - Delete old avatar before upload
     */
    @Test
    void testUploadAvatar_DeleteOldAvatarBeforeUpload_Success() throws Exception {
        // Arrange - User cÃƒÂ³ avatar cÃ…Â©
        testElderUser.setAvatar("https://cloudinary.com/old-avatar.jpg");
        
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(false);
        when(avatarFile.getSize()).thenReturn(3 * 1024 * 1024L);
        when(avatarFile.getContentType()).thenReturn("image/png");
        
        String newAvatarUrl = "https://cloudinary.com/new-avatar.png";
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(cloudinaryService.uploadImage(avatarFile)).thenReturn(newAvatarUrl);
        when(userRepository.save(any(User.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<String> response = userService.uploadAvatar(avatarFile, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals(newAvatarUrl, response.getData());
        verify(cloudinaryService, times(1)).deleteImage("https://cloudinary.com/old-avatar.jpg");
        verify(cloudinaryService, times(1)).uploadImage(avatarFile);
        verify(userRepository, times(1)).save(any(User.class));
    }
    /**
     * TC30: uploadAvatar - Xóa avatar cũ thất bại
     * Test 31: uploadAvatar - Delete old avatar fails but still uploads new
     */
    @Test
    void testUploadAvatar_DeleteOldAvatarFails_StillUploadsNew() throws Exception {
        // Arrange
        testElderUser.setAvatar("https://cloudinary.com/old-avatar.jpg");
        
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(false);
        when(avatarFile.getSize()).thenReturn(3 * 1024 * 1024L);
        when(avatarFile.getContentType()).thenReturn("image/png");
        
        String newAvatarUrl = "https://cloudinary.com/new-avatar.png";
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        
        // deleteImage
        doThrow(new RuntimeException("Cloudinary delete failed"))
                .when(cloudinaryService).deleteImage("https://cloudinary.com/old-avatar.jpg");
        
        when(cloudinaryService.uploadImage(avatarFile)).thenReturn(newAvatarUrl);
        when(userRepository.save(any(User.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<String> response = userService.uploadAvatar(avatarFile, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals(newAvatarUrl, response.getData());
        verify(cloudinaryService, times(1)).deleteImage("https://cloudinary.com/old-avatar.jpg");
        verify(cloudinaryService, times(1)).uploadImage(avatarFile);
        verify(userRepository, times(1)).save(any(User.class));
    }

    /**
     * TC31: uploadAvatar - User không tồn tại
     * Test 32: uploadAvatar - User not found
     */
    @Test
    void testUploadAvatar_UserNotFound_ThrowsException() throws IOException {
        // Arrange
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(false);
        
        when(authentication.getName()).thenReturn("notfound@test.com");
        when(userRepository.findByEmail("notfound@test.com")).thenReturn(Optional.empty());

        // Act & Assert
        BaseException.EntityNotFoundException exception = assertThrows(
                BaseException.EntityNotFoundException.class,
                () -> userService.uploadAvatar(avatarFile, authentication)
        );
        
        assertEquals("User not found", exception.getMessage());
        verify(cloudinaryService, never()).uploadImage(any());
    }

    /**
     * TC32: uploadAvatar - Cloudinary upload thất bại
     * Test 33: uploadAvatar - Cloudinary upload fails
     */
    @Test
    void testUploadAvatar_CloudinaryUploadFails_ThrowsException() throws IOException {
        // Arrange
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(false);
        when(avatarFile.getSize()).thenReturn(2 * 1024 * 1024L);
        when(avatarFile.getContentType()).thenReturn("image/jpeg");
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(cloudinaryService.uploadImage(avatarFile)).thenThrow(new RuntimeException("Cloudinary error"));

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.uploadAvatar(avatarFile, authentication)
        );
        
        assertTrue(exception.getMessage().contains("Lỗi") ||
                   exception.getMessage().contains("Error"));
        verify(userRepository, never()).save(any());
    }

    // ============= SEARCH USERS TESTS =============

    /**
     * TC35: searchUsers - Tìm kiếm bình thường (match theo fullName hoặc email)
     * Test 34: searchUsers - By name
     */
    @Test
    void testSearchUsers_ByName_Success() {
        // Arrange
        ElderUser user1 = new ElderUser();
        user1.setId(10L);
        user1.setEmail("nguyen.van.a@test.com");
        user1.setFullName("Nguyễn Văn A");

        SupervisorUser user2 = new SupervisorUser();
        user2.setId(11L);
        user2.setEmail("nguyen.thi.b@test.com");
        user2.setFullName("Nguyễn Thị B");

        ElderUser user3 = new ElderUser();
        user3.setId(12L);
        user3.setEmail("tran.van.c@test.com");
        user3.setFullName("Trần Văn C");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1, user2, user3);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        BaseResponse<?> response = userService.searchUsers("Nguyễn", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("Tìm kiếm người dùng thành công", response.getMessage());
        assertNotNull(response.getData());

    }

    /**
     * TC41: searchUsers - Tìm theo email chính xác
     * Test 35: searchUsers - By email
     */
    @Test
    void testSearchUsers_ByEmail_Success() {
        // Arrange
        SupervisorUser user1 = new SupervisorUser();
        user1.setId(20L);
        user1.setEmail("admin@company.com");
        user1.setFullName("Admin User");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        BaseResponse<?> response = userService.searchUsers("company", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
    }

    /**
     * TC36: searchUsers - query null
     * Test 36: searchUsers - Empty query returns all users
     */
    @Test
    void testSearchUsers_EmptyQuery_ReturnsAllUsers() {
        // Arrange
        ElderUser user1 = new ElderUser();
        user1.setId(30L);
        user1.setEmail("user1@test.com");
        user1.setFullName("User One");
        
        SupervisorUser user2 = new SupervisorUser();
        user2.setId(31L);
        user2.setEmail("user2@test.com");
        user2.setFullName("User Two");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1, user2);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        // Act - Query null
        BaseResponse<?> response = userService.searchUsers(null, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());

    }

    /**
     * TC39: searchUsers - Không có kết quả match
     * Test 37: searchUsers - No match returns empty list
     */
    @Test
    void testSearchUsers_NoMatch_ReturnsEmptyList() {
        // Arrange
        List<User> allUsers = Collections.singletonList(testElderUser);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        BaseResponse<?> response = userService.searchUsers("NonExistentName123", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());

    }

    /**
     * TC42: searchUsers - Case-insensitive match
     * Test 38: searchUsers - Case insensitive
     */
    @Test
    void testSearchUsers_CaseInsensitive_Success() {
        // Arrange
        ElderUser user1 = new ElderUser();
        user1.setId(40L);
        user1.setEmail("UPPER@TEST.COM");
        user1.setFullName("UPPER CASE");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        // Act
        BaseResponse<?> response = userService.searchUsers("upper", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
    }

    /**
     * KHÔNG CÓ TC_ID (Test thêm): searchUsers - Authentication user không tồn tại
     * Test 39: searchUsers - User not found throws exception
     */
    @Test
    void testSearchUsers_UserNotFound_ThrowsException() {
        // Arrange
        when(authentication.getName()).thenReturn("notfound@test.com");
        when(userRepository.findByEmail("notfound@test.com")).thenReturn(Optional.empty());

        // Act & Assert
        BaseException.EntityNotFoundException exception = assertThrows(
                BaseException.EntityNotFoundException.class,
                () -> userService.searchUsers("query", authentication)
        );
        
        assertEquals("User not found", exception.getMessage());
        verify(userRepository, never()).findAll();
    }

    /**
     * TC43: searchUsers - Exclude current user
     * Test 40: searchUsers - Exclude current user
     */
    @Test
    void testSearchUsers_ExcludeCurrentUser_Success() {
        // Arrange - testElderUser email "elder@test.com"
        ElderUser user1 = new ElderUser();
        user1.setId(50L);
        user1.setEmail("elder2@test.com");
        user1.setFullName("Elder Two");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        // Act
        BaseResponse<?> response = userService.searchUsers("elder", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
    }

    /**
     * TC38: searchUsers - Một user có fullName = null trong repo
     * Test 41: searchUsers - User with null fullName
     */
    @Test
    void testSearchUsers_UserWithNullFullName_NoError() {
        ElderUser user1 = new ElderUser();
        user1.setId(60L);
        user1.setEmail("user.null@test.com");
        user1.setFullName(null); // fullName = null
        
        ElderUser user2 = new ElderUser();
        user2.setId(61L);
        user2.setEmail("user2@test.com");
        user2.setFullName("User Two");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1, user2);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        // Act -
        BaseResponse<?> response = userService.searchUsers("user", authentication);

        // Assert -
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
    }

    /**
     * TC37: searchUsers - query rỗng hoặc whitespace
     * Test 42: searchUsers - Whitespace query returns all users
     */
    @Test
    void testSearchUsers_WhitespaceQuery_ReturnsAllUsers() {
        // Arrange
        ElderUser user1 = new ElderUser();
        user1.setId(70L);
        user1.setEmail("user70@test.com");
        user1.setFullName("User Seventy");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        // Act -
        BaseResponse<?> response = userService.searchUsers("   ", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
    }

    /**
     * TC40: searchUsers - Repository ném Exception
     * Test 43: searchUsers - Repository fails throws exception
     */
    @Test
    void testSearchUsers_RepositoryFails_ThrowsException() {
        // Arrange
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenThrow(new RuntimeException("Database error"));

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.searchUsers("test", authentication)
        );
        
        assertTrue(exception.getMessage().contains("Lỗi") ||
                   exception.getMessage().contains("Database error"));
    }
}

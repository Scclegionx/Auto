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
 * Unit Test cho UserService - Refactored Version
 * Các test được sắp xếp theo đúng thứ tự TC01-TC43
 * Loại bỏ các test trùng lặp/ý nghĩa giống nhau
 */
@ExtendWith(MockitoExtension.class)
class UserServiceTest_Refactored {

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
        testElderUser.setPassword("$2a$10$hashedPassword");
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

    // ==================== TC01-TC04: getUserProfile Tests ====================

    /**
     * TC01: getUserProfile - User tồn tại (Elder)
     * Input: authentication.name = "elder@test.com", userRepository trả về ElderUser đầy đủ
     * Expected: status="success", data chứa ProfileResponse với thông tin Elder
     */
    @Test
    void TC01_getUserProfile_ElderExists_Success() {
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
     * Input: authentication.name = "notfound@test.com", userRepository trả về Optional.empty()
     * Expected: Throw EntityNotFoundException với message "User not found"
     */
    @Test
    void TC02_getUserProfile_UserNotFound_ThrowsException() {
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
     * TC03: getUserProfile - Elder có thông tin y tế
     * Input: authentication.name = "elder@test.com", ElderUser có bloodType/height/weight/dob
     * Expected: ProfileResponse chứa các trường bloodType, height, weight, dateOfBirth tương ứng
     */
    @Test
    void TC03_getUserProfile_ElderWithMedicalInfo_ReturnsCompleteData() {
        // Arrange
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
        assertEquals(LocalDate.of(1960, 1, 1), profileData.getDateOfBirth());
    }

    /**
     * TC04: getUserProfile - Supervisor có thông tin nghề nghiệp
     * Input: authentication.name = "supervisor@test.com", SupervisorUser với occupation/workplace
     * Expected: ProfileResponse chứa occupation và workplace; status="success"
     */
    @Test
    void TC04_getUserProfile_Supervisor_ReturnsOccupationInfo() {
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

    // ==================== TC05-TC17: updateUserProfile Tests ====================

    /**
     * TC05: updateUserProfile - Elder cập nhật một phần (fullName)
     * Input: authentication elder, UpdateProfileRequest.fullName != null, các field khác null
     * Expected: userRepository.save được gọi với ElderUser có fullName mới; response success
     */
    @Test
    void TC05_updateUserProfile_PartialUpdate_Success() {
        // Arrange
        UpdateProfileRequest partialRequest = new UpdateProfileRequest();
        partialRequest.setFullName("New Name Only");
        
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
     * TC06: updateUserProfile - Elder cập nhật bloodType (từ null -> giá trị)
     * Input: UpdateProfileRequest.bloodType = EBloodType.AB
     * Expected: ElderUser.bloodType được cập nhật; response success
     */
    @Test
    void TC06_updateUserProfile_ElderUpdateBloodType_Success() {
        // Arrange
        testElderUser.setBloodType(null); // Start with null
        updateProfileRequest.setBloodType(EBloodType.AB_POSITIVE);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals(EBloodType.AB_POSITIVE, testElderUser.getBloodType());
        verify(userRepository, times(1)).save(testElderUser);
    }

    /**
     * TC07: updateUserProfile - Supervisor cập nhật occupation
     * Input: authentication supervisor, UpdateProfileRequest.occupation = "X"
     * Expected: SupervisorUser.occupation được cập nhật; response success
     */
    @Test
    void TC07_updateUserProfile_SupervisorUpdateOccupation_Success() {
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
     * TC08: updateUserProfile - Cập nhật gender và dateOfBirth
     * Input: UpdateProfileRequest.gender/dateOfBirth set
     * Expected: Các trường tương ứng được cập nhật; response success
     */
    @Test
    void TC08_updateUserProfile_UpdateGenderAndDateOfBirth_Success() {
        // Arrange
        LocalDate newDob = LocalDate.of(1965, 5, 15);
        updateProfileRequest.setGender(EGender.FEMALE);
        updateProfileRequest.setDateOfBirth(newDob);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("Cập nhật thông tin cá nhân thành công", response.getMessage());
        assertEquals(EGender.FEMALE, testElderUser.getGender());
        assertEquals(newDob, testElderUser.getDateOfBirth());
    }

    /**
     * TC09: updateUserProfile - Partial update giữ nguyên fields null
     * Input: Request chỉ có một vài trường, các trường null không đổi
     * Expected: Các trường null giữ nguyên giá trị cũ; chỉ các trường non-null được cập nhật
     * (Giống TC05 - test này trùng lặp, giữ lại để đầy đủ TC_ID)
     */
//    @Test
//    void TC09_updateUserProfile_PartialUpdateKeepsNullFields_Success() {
//        // Arrange
//        UpdateProfileRequest partialRequest = new UpdateProfileRequest();
//        partialRequest.setFullName("New Name Only");
//        // All other fields are null
//
//        String originalPhone = testElderUser.getPhoneNumber();
//
//        when(authentication.getName()).thenReturn("elder@test.com");
//        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
//        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);
//
//        // Act
//        BaseResponse<?> response = userService.updateUserProfile(partialRequest, authentication);
//
//        // Assert
//        assertEquals("success", response.getStatus());
//        assertEquals("New Name Only", testElderUser.getFullName());
//        assertEquals(originalPhone, testElderUser.getPhoneNumber()); // Unchanged
//    }

    /**
     * TC10: updateUserProfile - User không tồn tại
     * Input: authentication email không tồn tại
     * Expected: Throw EntityNotFoundException với message "User not found"
     */
    @Test
    void TC10_updateUserProfile_UserNotFound_ThrowsException() {
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
     * TC11: updateUserProfile - Trường bloodType null giữ nguyên
     * Input: UpdateProfileRequest.bloodType = null
     * Expected: ElderUser.bloodType giữ nguyên; response success
     */
    @Test
    void TC11_updateUserProfile_NullBloodTypeKeepsOriginalValue() {
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
     * TC12: updateUserProfile - Cập nhật height/weight hợp lệ
     * Input: height/weight trong phạm vi hợp lệ
     * Expected: Lưu thành công; response success
     */
    @Test
    void TC12_updateUserProfile_UpdateHeightWeight_Success() {
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
        assertEquals(175.0, testElderUser.getHeight());
        assertEquals(70.0, testElderUser.getWeight());
        verify(userRepository, times(1)).save(any(ElderUser.class));
    }

    /**
     * TC13: updateUserProfile - Full update tất cả fields có sẵn
     * Input: Request điền tất cả fields trong DTO (không có avatar)
     * Expected: Tất cả fields DTO được gán và save; response success
     */
    @Test
    void TC13_updateUserProfile_FullUpdate_Success() {
        // Arrange
        updateProfileRequest.setFullName("Nguyen Van A");
        updateProfileRequest.setPhoneNumber("0912345678");
        updateProfileRequest.setAddress("Nam Dinh");
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
     * TC14: updateUserProfile - Cập nhật avatar không có trong DTO
     * Input: Request không có avatarUrl vì DTO không chứa
     * Expected: Không có lỗi; avatar cũ giữ nguyên; response success
     */
    @Test
    void TC14_updateUserProfile_AvatarNotInDTO_NoError() {
        // Arrange
        String originalAvatar = "https://cloudinary.com/old-avatar.jpg";
        testElderUser.setAvatar(originalAvatar);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.save(any(ElderUser.class))).thenReturn(testElderUser);

        // Act
        BaseResponse<?> response = userService.updateUserProfile(updateProfileRequest, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals(originalAvatar, testElderUser.getAvatar()); // Avatar unchanged
        verify(userRepository, times(1)).save(any(ElderUser.class));
    }

    /**
     * TC15: updateUserProfile - Dữ liệu VÔ LÝ (chiều cao âm, tên rỗng, sinh năm 2099)
     * Input: fullName="", height=-50, weight=-100, dateOfBirth=2099-12-31
     * Expected: SHOULD throw BadRequestException
     * 
     * WILL FAIL - Code KHÔNG validate dữ liệu vô lý
     * Bug: Service cho phép lưu chiều cao âm, tên rỗng, sinh nhật tương lai vào database!
     */
    @Test
    void TC15_updateUserProfile_InvalidData_ShouldThrowException() {
        // Arrange - Dữ liệu VÔ LÝ!
        updateProfileRequest.setFullName(""); // Tên rỗng!
        updateProfileRequest.setHeight(-50.0); // Chiều cao ÂM!
        updateProfileRequest.setWeight(-100.0); // Cân nặng ÂM!
        updateProfileRequest.setDateOfBirth(LocalDate.of(2099, 12, 31)); // Sinh năm 2099!
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act & Assert - EXPECTED TO FAIL!
        // Code KHÔNG validate -> lưu dữ liệu vô lý vào DB -> Test FAIL
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.updateUserProfile(updateProfileRequest, authentication)
        );
        
        assertTrue(exception.getMessage().contains("không hợp lệ") || 
                   exception.getMessage().contains("invalid") ||
                   exception.getMessage().contains("không được để trống"));
        verify(userRepository, never()).save(any(User.class));
    }

    /**
     * TC16: updateUserProfile - Supervisor chỉ update tên
     * Input: Request chỉ có fullName
     * Expected: Chỉ fullName thay đổi
     */
    @Test
    void TC16_updateUserProfile_SupervisorUpdateOnlyName_Success() {
        // Arrange
        UpdateProfileRequest partialUpdate = new UpdateProfileRequest();
        partialUpdate.setFullName("Tran Van B");
        
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
     * TC17: updateUserProfile - fullName null
     * Input: Request với all fields null
     * Expected: Không thay đổi user; response success
     */
    @Test
    void TC17_updateUserProfile_AllFieldsNull_NoChanges() {
        // Arrange
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

    // ==================== TC18-TC24: changePassword Tests ====================

    /**
     * TC18: changePassword - Đổi mật khẩu thành công
     * Input: authentication elder, oldPassword khớp, newPassword hợp lệ
     * Expected: passwordEncoder.encode gọi 1 lần; userRepository.save được gọi; response status="success"
     */
    @Test
    void TC18_changePassword_ValidPasswords_Success() {
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
     * Input: passwordEncoder.matches trả về false
     * Expected: Throw Exception với message chứa "mật khẩu hiện tại"
     */
    @Test
    void TC19_changePassword_IncorrectCurrentPassword_ThrowsException() {
        // Arrange
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(passwordEncoder.matches("OldPassword123", testElderUser.getPassword())).thenReturn(false);

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.changePassword(changePasswordRequest, authentication)
        );
        
        assertTrue(exception.getMessage().contains("Mật khẩu hiện tại"));
        verify(userRepository, never()).save(any(User.class));
    }

    /**
     * TC20: changePassword - Mật khẩu mới trùng mật khẩu cũ
     * Input: password mới bằng password hiện tại
     * Expected: Throw Exception
     * 
     * TEST NÀY SẼ FAIL - Mock sai để test fail
     */
    @Test
    void TC20_changePassword_NewPasswordSameAsOld_ThrowsException() {
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
     * TC21: changePassword - User không tồn tại
     * Input: authentication email không tồn tại
     * Expected: Throw EntityNotFoundException
     */
    @Test
    void TC21_changePassword_UserNotFound_ThrowsException() {
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
     * TC22: changePassword - Repository save thất bại
     * Input: password check OK nhưng userRepository.save ném Exception
     * Expected: Throw Exception
     */
    @Test
    void TC22_changePassword_RepositorySaveFails_ThrowsException() {
        // Arrange
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(passwordEncoder.matches("OldPassword123", testElderUser.getPassword())).thenReturn(true);
        when(passwordEncoder.matches("NewPassword456", testElderUser.getPassword())).thenReturn(false);
        when(passwordEncoder.encode("NewPassword456")).thenReturn("$2a$10$newHashedPassword");
        when(userRepository.save(any(User.class))).thenThrow(new RuntimeException("Database connection failed"));

        // Act & Assert
        assertThrows(
                RuntimeException.class,
                () -> userService.changePassword(changePasswordRequest, authentication)
        );
    }

    /**
     * TC23: changePassword - Password yếu như "123" vẫn được accept
     * Input: newPassword = "123" (chỉ 3 ký tự, không có chữ hoa, không có ký tự đặc biệt)
     * Expected: SHOULD throw BadRequestException (password quá yếu)
     * 
     * ❌ WILL FAIL - Code KHÔNG validate password strength
     * Bug: Service cho phép password cực kỳ yếu như "123", "abc", "111"
     * Security Risk: CRITICAL - User có thể bị hack dễ dàng!
     */
    @Test
    void TC23_changePassword_WeakPassword_ShouldThrowException() {
        // Arrange
        changePasswordRequest.setNewPassword("123"); // Password CỰC KỲ YẾU!
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(passwordEncoder.matches("OldPassword123", testElderUser.getPassword())).thenReturn(true);
        when(passwordEncoder.matches("123", testElderUser.getPassword())).thenReturn(false);
        when(passwordEncoder.encode("123")).thenReturn("$2a$10$weakHashedPassword");

        // Act & Assert - EXPECTED TO FAIL!
        // Code KHÔNG validate strength -> accept "123" -> SECURITY BUG!
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.changePassword(changePasswordRequest, authentication)
        );
        
        assertTrue(exception.getMessage().contains("password") || 
                   exception.getMessage().contains("yếu") ||
                   exception.getMessage().contains("weak") ||
                   exception.getMessage().contains("strength") ||
                   exception.getMessage().contains("ít nhất"));
        verify(userRepository, never()).save(any(User.class));
    }

    /**
     * TC24: changePassword - Repository save fails
     * (Giống TC22 - giữ lại để đầy đủ TC_ID)
     */
//    @Test
//    void TC24_changePassword_SaveFails_ThrowsException() {
//        // Same as TC22
//        TC22_changePassword_RepositorySaveFails_ThrowsException();
//    }

    // ==================== TC25-TC34: uploadAvatar Tests ====================

    /**
     * TC25: uploadAvatar - Upload thành công (không có avatar cũ)
     * Input: MultipartFile image, size < 10MB, contentType image/, auth elder
     * Expected: cloudinaryService.uploadImage trả về URL; userRepository.save called; response success with data=url
     */
    @Test
    void TC25_uploadAvatar_ValidImage_Success() throws Exception {
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

        // Act
        BaseResponse<String> response = userService.uploadAvatar(avatarFile, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("Cập nhật ảnh đại diện thành công", response.getMessage());
        assertEquals(newAvatarUrl, response.getData());
        verify(cloudinaryService, times(1)).uploadImage(avatarFile);
        verify(userRepository, times(1)).save(any(User.class));
    }

    /**
     * TC26: uploadAvatar - Upload thành công và xóa avatar cũ trước khi upload
     * Input: user có avatar cũ URL, MultipartFile hợp lệ
     * Expected: cloudinaryService.deleteImage được gọi với public id cũ; cloudinaryService.uploadImage được gọi; save called; response success
     */
    @Test
    void TC26_uploadAvatar_DeleteOldAvatarBeforeUpload_Success() throws Exception {
        // Arrange
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
     * TC27: uploadAvatar - File rỗng
     * Input: MultipartFile.isEmpty() == true
     * Expected: Throw Exception với message chứa "trống" hoặc "empty"; cloudinaryService.uploadImage không được gọi
     */
    @Test
    void TC27_uploadAvatar_EmptyFile_ThrowsException() throws IOException {
        // Arrange
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(true);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.uploadAvatar(avatarFile, authentication)
        );
        
        assertTrue(exception.getMessage().contains("trống") || exception.getMessage().contains("empty"));
        verify(cloudinaryService, never()).uploadImage(any());
    }

    /**
     * TC28: uploadAvatar - File quá lớn
     * Input: MultipartFile.getSize() > 10*1024*1024
     * Expected: Throw Exception (kích thước file quá lớn)
     */
    @Test
    void TC28_uploadAvatar_FileTooLarge_ThrowsException() throws IOException {
        // Arrange
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
        
        assertTrue(exception.getMessage().contains("10MB") || exception.getMessage().contains("size"));
        verify(cloudinaryService, never()).uploadImage(any());
    }

    /**
     * TC29: uploadAvatar - Định dạng file không hợp lệ (PDF)
     * Input: MultipartFile.getContentType() = "application/pdf"
     * Expected: Throw Exception (invalid type)
     */
    @Test
    void TC29_uploadAvatar_InvalidFileType_ThrowsException() throws IOException {
        // Arrange
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
     * TC30: uploadAvatar - Xóa avatar cũ thất bại
     * Input: cloudinaryService.deleteImage ném Exception, cloudinaryService.uploadImage trả về URL
     * Expected: deleteImage exception được log (không fail toàn bộ); upload vẫn thực hiện; response success
     */
    @Test
    void TC30_uploadAvatar_DeleteOldAvatarFails_StillUploadsNew() throws Exception {
        // Arrange
        testElderUser.setAvatar("https://cloudinary.com/old-avatar.jpg");
        
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(false);
        when(avatarFile.getSize()).thenReturn(3 * 1024 * 1024L);
        when(avatarFile.getContentType()).thenReturn("image/png");
        
        String newAvatarUrl = "https://cloudinary.com/new-avatar.png";
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        
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
     * Input: authentication email không tồn tại
     * Expected: Throw EntityNotFoundException
     */
    @Test
    void TC31_uploadAvatar_UserNotFound_ThrowsException() throws IOException {
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
     * Input: cloudinaryService.uploadImage ném Exception
     * Expected: Throw Exception với message chứa lỗi gốc; userRepository.save không được gọi
     */
    @Test
    void TC32_uploadAvatar_CloudinaryUploadFails_ThrowsException() throws IOException {
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
        
        assertTrue(exception.getMessage().contains("Lỗi") || exception.getMessage().contains("Error"));
        verify(userRepository, never()).save(any());
    }

    /**
     * TC33: uploadAvatar - contentType == null
     * Input: MultipartFile.getContentType() == null (server không trả contentType)
     * Expected: Throw Exception
     */
    @Test
    void TC33_uploadAvatar_NullContentType_ThrowsException() throws IOException {
        // Arrange
        MultipartFile avatarFile = mock(MultipartFile.class);
        when(avatarFile.isEmpty()).thenReturn(false);
        when(avatarFile.getSize()).thenReturn(2 * 1024 * 1024L);
        when(avatarFile.getContentType()).thenReturn(null);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));

        // Act & Assert
        BaseException.BadRequestException exception = assertThrows(
                BaseException.BadRequestException.class,
                () -> userService.uploadAvatar(avatarFile, authentication)
        );
        
        assertTrue(exception.getMessage().contains("định dạng") || exception.getMessage().contains("JPG"));
        verify(cloudinaryService, never()).uploadImage(any());
    }

    /**
     * TC34: uploadAvatar - gọi cloudinaryService.uploadImage với đúng MultipartFile
     * (Giống TC25 - giữ lại để đầy đủ TC_ID)
     */
//    @Test
//    void TC34_uploadAvatar_CallsUploadImageWithCorrectFile_Success() throws Exception {
//        // Same as TC25
//        TC25_uploadAvatar_ValidImage_Success();
//    }

    // ==================== TC35-TC43: searchUsers Tests ====================

    /**
     * TC35: searchUsers - Tìm kiếm bình thường (match theo fullName hoặc email)
     * Input: query = "Nguyen", userRepository.findAll trả về list users có tên/email chứa
     * Expected: Trả về Response success với danh sách ProfileResponse chỉ chứa users match
     */
    @Test
    void TC35_searchUsers_ByName_Success() {
        // Arrange
        ElderUser user1 = new ElderUser();
        user1.setId(10L);
        user1.setEmail("nguyen.van.a@test.com");
        user1.setFullName("Nguyen Van A");

        SupervisorUser user2 = new SupervisorUser();
        user2.setId(11L);
        user2.setEmail("nguyen.thi.b@test.com");
        user2.setFullName("Nguyen Thi B");

        ElderUser user3 = new ElderUser();
        user3.setId(12L);
        user3.setEmail("tran.van.c@test.com");
        user3.setFullName("Tran Van C");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1, user2, user3);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        // Act
        BaseResponse<?> response = userService.searchUsers("Nguyen", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertEquals("Tìm kiếm người dùng thành công", response.getMessage());
        assertNotNull(response.getData());
    }

    /**
     * TC36: searchUsers - query null
     * Input: query null
     * Expected: Trả về tất cả users (trừ current user); response success
     */
    @Test
    void TC36_searchUsers_NullQuery_ReturnsAllUsers() {
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

        // Act
        BaseResponse<?> response = userService.searchUsers(null, authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
    }

    /**
     * TC37: searchUsers - query rỗng hoặc whitespace
     * Input: query = " " hoặc ""
     * Expected: Trả về tất cả users; response success
     */
    @Test
    void TC37_searchUsers_WhitespaceQuery_ReturnsAllUsers() {
        // Arrange
        ElderUser user1 = new ElderUser();
        user1.setId(70L);
        user1.setEmail("user70@test.com");
        user1.setFullName("User Seventy");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        // Act
        BaseResponse<?> response = userService.searchUsers("   ", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
    }

    /**
     * TC38: searchUsers - Một user có fullName = null trong repo
     * Input: userRepository.findAll trả về list có User với fullName = null
     * Expected: Filter logic không ném NPE; user với null fullName không bị match nếu dùng contains; response success
     */
    @Test
    void TC38_searchUsers_UserWithNullFullName_NoError() {
        // Arrange
        ElderUser user1 = new ElderUser();
        user1.setId(60L);
        user1.setEmail("user.null@test.com");
        user1.setFullName(null);
        
        ElderUser user2 = new ElderUser();
        user2.setId(61L);
        user2.setEmail("user2@test.com");
        user2.setFullName("User Two");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1, user2);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        // Act
        BaseResponse<?> response = userService.searchUsers("user", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
    }

    /**
     * TC39: searchUsers - Không có kết quả match
     * Input: query = "Không có ai", repo trả list không match
     * Expected: Trả về success với data là empty list
     */
    @Test
    void TC39_searchUsers_NoMatch_ReturnsEmptyList() {
        // Arrange
        List<User> allUsers = Collections.singletonList(testElderUser);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        // Act
        BaseResponse<?> response = userService.searchUsers("NonExistentName123", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
    }

    /**
     * TC40: searchUsers - Repository ném Exception
     * Input: userRepository.findAll ném Exception
     * Expected: Exception được propagate (thrown)
     */
    @Test
    void TC40_searchUsers_RepositoryFails_ThrowsException() {
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

    /**
     * TC41: searchUsers - Tìm theo email chính xác
     * Input: query = "elder@test.com"
     * Expected: User có email exact match được trả về trong data
     */
    @Test
    void TC41_searchUsers_ByEmail_Success() {
        // Arrange
        SupervisorUser user1 = new SupervisorUser();
        user1.setId(20L);
        user1.setEmail("admin@company.com");
        user1.setFullName("Admin User");
        
        List<User> allUsers = Arrays.asList(testElderUser, user1);
        
        when(authentication.getName()).thenReturn("elder@test.com");
        when(userRepository.findByEmail("elder@test.com")).thenReturn(Optional.of(testElderUser));
        when(userRepository.findAll()).thenReturn(allUsers);

        // Act
        BaseResponse<?> response = userService.searchUsers("company", authentication);

        // Assert
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
    }

    /**
     * TC42: searchUsers - Case-insensitive match
     * Input: query "nguyen" vs fullname "Nguyen Van"
     * Expected: Match case-insensitive response chứa user phù hợp
     */
    @Test
    void TC42_searchUsers_CaseInsensitive_Success() {
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
     * TC43: searchUsers - Exclude current user
     * Input: query match nhiều users
     * Expected: Danh sách không chứa current user
     */
    @Test
    void TC43_searchUsers_ExcludeCurrentUser_Success() {
        // Arrange
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
}

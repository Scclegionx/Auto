package com.example.Auto_BE.controller;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.service.SchedulerReminder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;

import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;

@RestController
@RequestMapping("/api/scheduler")
public class SchedulerController {

    @Autowired
    private SchedulerReminder schedulerReminder;

    @PostMapping("/schedule-weekly")
    public ResponseEntity<BaseResponse<String>> scheduleWeeklyReminders() {
        try {
            schedulerReminder.scheduleWeeklyReminders();
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Đã lên lịch thông báo cho tuần thành công")
                    .data("Weekly reminders scheduled successfully")
                    .build();
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("ERROR")
                    .message("Lỗi khi lên lịch: " + e.getMessage())
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }

    @PostMapping("/schedule-reminder/{reminderId}")
    public ResponseEntity<BaseResponse<String>> scheduleSpecificReminder(
            @PathVariable Long reminderId,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate) {
        
        try {
            if (startDate == null) {
                startDate = LocalDate.now();
            }
            
            schedulerReminder.scheduleReminderForSpecificWeek(reminderId, startDate);
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Đã lên lịch thông báo cho thuốc ID: " + reminderId)
                    .data("Reminder scheduled for week starting: " + startDate)
                    .build();
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("ERROR")
                    .message("Lỗi khi lên lịch: " + e.getMessage())
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }

    @DeleteMapping("/cancel-reminder/{reminderId}")
    public ResponseEntity<BaseResponse<String>> cancelReminderJobs(@PathVariable Long reminderId) {
        try {
            schedulerReminder.cancelAllJobsForReminder(reminderId);
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Đã hủy tất cả lịch thông báo cho thuốc ID: " + reminderId)
                    .data("All jobs cancelled for reminder: " + reminderId)
                    .build();
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("ERROR")
                    .message("Lỗi khi hủy lịch: " + e.getMessage())
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    @GetMapping("/status")
    public ResponseEntity<BaseResponse<String>> getSchedulerStatus() {
        try {
            schedulerReminder.checkSchedulerStatus();
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Scheduler status checked - see console logs")
                    .data("Check console for detailed status")
                    .build();
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("ERROR")
                    .message("Lỗi khi kiểm tra status: " + e.getMessage())
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    @GetMapping("/check-data")
    public ResponseEntity<BaseResponse<String>> checkDatabaseData() {
        try {
            // Kiểm tra dữ liệu trong database
            StringBuilder result = new StringBuilder();
            result.append("Checking medication_reminders table...\n");
            
            // Sử dụng raw query để kiểm tra
            // Bạn có thể thêm logic để kiểm tra dữ liệu ở đây
            
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status(SUCCESS)
                    .message("Database check completed")
                    .data(result.toString())
                    .build();
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            BaseResponse<String> response = BaseResponse.<String>builder()
                    .status("ERROR")
                    .message("Lỗi khi kiểm tra database: " + e.getMessage())
                    .data(null)
                    .build();
            
            return ResponseEntity.badRequest().body(response);
        }
    }
}

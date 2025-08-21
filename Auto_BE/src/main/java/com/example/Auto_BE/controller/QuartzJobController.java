//package com.example.Auto_BE.controller;
//
//import com.example.Auto_BE.dto.BaseResponse;
//import org.quartz.*;
//import org.quartz.impl.matchers.GroupMatcher;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.http.ResponseEntity;
//import org.springframework.web.bind.annotation.*;
//
//import java.time.LocalDateTime;
//import java.time.ZoneId;
//import java.util.*;
//
//import static com.example.Auto_BE.constants.SuccessMessage.SUCCESS;
//
//@RestController
//@RequestMapping("/api/quartz")
//public class QuartzJobController {
//
//    @Autowired
//    private Scheduler scheduler;
//
//    @GetMapping("/jobs")
//    public ResponseEntity<BaseResponse<Map<String, Object>>> getAllJobs() {
//        try {
//            Map<String, Object> result = new HashMap<>();
//            List<Map<String, Object>> jobsList = new ArrayList<>();
//
//            // Lấy thông tin tổng quan
//            Map<String, Object> schedulerInfo = new HashMap<>();
//            schedulerInfo.put("schedulerName", scheduler.getSchedulerName());
//            schedulerInfo.put("instanceId", scheduler.getSchedulerInstanceId());
//            schedulerInfo.put("isStarted", scheduler.isStarted());
//            schedulerInfo.put("isInStandby", scheduler.isInStandbyMode());
//            schedulerInfo.put("isShutdown", scheduler.isShutdown());
//
//            int totalJobs = 0;
//            int runningJobs = 0;
//            int pausedJobs = 0;
//
//            // Lấy tất cả jobs
//            for (String groupName : scheduler.getJobGroupNames()) {
//                for (JobKey jobKey : scheduler.getJobKeys(GroupMatcher.jobGroupEquals(groupName))) {
//                    JobDetail jobDetail = scheduler.getJobDetail(jobKey);
//                    List<? extends Trigger> triggers = scheduler.getTriggersOfJob(jobKey);
//
//                    Map<String, Object> jobInfo = new HashMap<>();
//                    jobInfo.put("jobName", jobKey.getName());
//                    jobInfo.put("jobGroup", jobKey.getGroup());
//                    jobInfo.put("jobClass", jobDetail.getJobClass().getSimpleName());
//                    jobInfo.put("description", jobDetail.getDescription());
//                    jobInfo.put("isDurable", jobDetail.isDurable());
//                    jobInfo.put("requestsRecovery", jobDetail.requestsRecovery());
//
//                    // Thông tin JobData
//                    JobDataMap dataMap = jobDetail.getJobDataMap();
//                    Map<String, Object> jobData = new HashMap<>();
//                    for (String key : dataMap.getKeys()) {
//                        jobData.put(key, dataMap.get(key));
//                    }
//                    jobInfo.put("jobData", jobData);
//
//                    // Thông tin Triggers
//                    List<Map<String, Object>> triggersList = new ArrayList<>();
//                    for (Trigger trigger : triggers) {
//                        Map<String, Object> triggerInfo = new HashMap<>();
//                        triggerInfo.put("triggerName", trigger.getKey().getName());
//                        triggerInfo.put("triggerGroup", trigger.getKey().getGroup());
//                        triggerInfo.put("triggerClass", trigger.getClass().getSimpleName());
//                        triggerInfo.put("description", trigger.getDescription());
//
//                        // Thời gian
//                        if (trigger.getNextFireTime() != null) {
//                            triggerInfo.put("nextFireTime",
//                                    LocalDateTime.ofInstant(trigger.getNextFireTime().toInstant(), ZoneId.systemDefault()));
//                        }
//                        if (trigger.getPreviousFireTime() != null) {
//                            triggerInfo.put("previousFireTime",
//                                    LocalDateTime.ofInstant(trigger.getPreviousFireTime().toInstant(), ZoneId.systemDefault()));
//                        }
//                        triggerInfo.put("startTime",
//                                LocalDateTime.ofInstant(trigger.getStartTime().toInstant(), ZoneId.systemDefault()));
//                        if (trigger.getEndTime() != null) {
//                            triggerInfo.put("endTime",
//                                    LocalDateTime.ofInstant(trigger.getEndTime().toInstant(), ZoneId.systemDefault()));
//                        }
//
//                        // Trạng thái
//                        Trigger.TriggerState state = scheduler.getTriggerState(trigger.getKey());
//                        triggerInfo.put("state", state.name());
//
//                        if (state == Trigger.TriggerState.NORMAL) runningJobs++;
//                        else if (state == Trigger.TriggerState.PAUSED) pausedJobs++;
//
//                        triggersList.add(triggerInfo);
//                    }
//
//                    jobInfo.put("triggers", triggersList);
//                    jobsList.add(jobInfo);
//                    totalJobs++;
//                }
//            }
//
//            schedulerInfo.put("totalJobs", totalJobs);
//            schedulerInfo.put("runningJobs", runningJobs);
//            schedulerInfo.put("pausedJobs", pausedJobs);
//
//            result.put("schedulerInfo", schedulerInfo);
//            result.put("jobs", jobsList);
//
//            BaseResponse<Map<String, Object>> response = BaseResponse.<Map<String, Object>>builder()
//                    .status(SUCCESS)
//                    .message("Lấy danh sách jobs thành công")
//                    .data(result)
//                    .build();
//
//            return ResponseEntity.ok(response);
//
//        } catch (Exception e) {
//            BaseResponse<Map<String, Object>> response = BaseResponse.<Map<String, Object>>builder()
//                    .status("ERROR")
//                    .message("Lỗi khi lấy danh sách jobs: " + e.getMessage())
//                    .data(null)
//                    .build();
//
//            return ResponseEntity.badRequest().body(response);
//        }
//    }
//
//    @GetMapping("/jobs/{jobName}")
//    public ResponseEntity<BaseResponse<Map<String, Object>>> getJobDetail(
//            @PathVariable String jobName,
//            @RequestParam(defaultValue = "DEFAULT") String jobGroup) {
//
//        try {
//            JobKey jobKey = new JobKey(jobName, jobGroup);
//
//            if (!scheduler.checkExists(jobKey)) {
//                BaseResponse<Map<String, Object>> response = BaseResponse.<Map<String, Object>>builder()
//                        .status("ERROR")
//                        .message("Job không tồn tại")
//                        .data(null)
//                        .build();
//                return ResponseEntity.notFound().build();
//            }
//
//            JobDetail jobDetail = scheduler.getJobDetail(jobKey);
//            List<? extends Trigger> triggers = scheduler.getTriggersOfJob(jobKey);
//
//            Map<String, Object> jobInfo = new HashMap<>();
//            jobInfo.put("jobName", jobKey.getName());
//            jobInfo.put("jobGroup", jobKey.getGroup());
//            jobInfo.put("jobClass", jobDetail.getJobClass().getName());
//            jobInfo.put("description", jobDetail.getDescription());
//            jobInfo.put("isDurable", jobDetail.isDurable());
//            jobInfo.put("requestsRecovery", jobDetail.requestsRecovery());
//
//            // JobData detail
//            JobDataMap dataMap = jobDetail.getJobDataMap();
//            Map<String, Object> jobData = new HashMap<>();
//            for (String key : dataMap.getKeys()) {
//                jobData.put(key, dataMap.get(key));
//            }
//            jobInfo.put("jobData", jobData);
//
//            // Triggers detail
//            List<Map<String, Object>> triggersList = new ArrayList<>();
//            for (Trigger trigger : triggers) {
//                Map<String, Object> triggerInfo = new HashMap<>();
//                triggerInfo.put("triggerName", trigger.getKey().getName());
//                triggerInfo.put("triggerGroup", trigger.getKey().getGroup());
//                triggerInfo.put("triggerClass", trigger.getClass().getName());
//                triggerInfo.put("description", trigger.getDescription());
//                triggerInfo.put("priority", trigger.getPriority());
//                triggerInfo.put("misfireInstruction", trigger.getMisfireInstruction());
//
//                // Thời gian chi tiết
//                if (trigger.getNextFireTime() != null) {
//                    triggerInfo.put("nextFireTime",
//                            LocalDateTime.ofInstant(trigger.getNextFireTime().toInstant(), ZoneId.systemDefault()));
//                }
//                if (trigger.getPreviousFireTime() != null) {
//                    triggerInfo.put("previousFireTime",
//                            LocalDateTime.ofInstant(trigger.getPreviousFireTime().toInstant(), ZoneId.systemDefault()));
//                }
//                triggerInfo.put("startTime",
//                        LocalDateTime.ofInstant(trigger.getStartTime().toInstant(), ZoneId.systemDefault()));
//                if (trigger.getEndTime() != null) {
//                    triggerInfo.put("endTime",
//                            LocalDateTime.ofInstant(trigger.getEndTime().toInstant(), ZoneId.systemDefault()));
//                }
//
//                // Trạng thái và thống kê
//                Trigger.TriggerState state = scheduler.getTriggerState(trigger.getKey());
//                triggerInfo.put("state", state.name());
//                triggerInfo.put("mayFireAgain", trigger.mayFireAgain());
//
//                triggersList.add(triggerInfo);
//            }
//
//            jobInfo.put("triggers", triggersList);
//
//            BaseResponse<Map<String, Object>> response = BaseResponse.<Map<String, Object>>builder()
//                    .status(SUCCESS)
//                    .message("Lấy thông tin job thành công")
//                    .data(jobInfo)
//                    .build();
//
//            return ResponseEntity.ok(response);
//
//        } catch (Exception e) {
//            BaseResponse<Map<String, Object>> response = BaseResponse.<Map<String, Object>>builder()
//                    .status("ERROR")
//                    .message("Lỗi khi lấy thông tin job: " + e.getMessage())
//                    .data(null)
//                    .build();
//
//            return ResponseEntity.badRequest().body(response);
//        }
//    }
//
//    @GetMapping("/jobs/running")
//    public ResponseEntity<BaseResponse<List<Map<String, Object>>>> getCurrentlyExecutingJobs() {
//        try {
//            List<JobExecutionContext> executingJobs = scheduler.getCurrentlyExecutingJobs();
//            List<Map<String, Object>> result = new ArrayList<>();
//
//            for (JobExecutionContext context : executingJobs) {
//                Map<String, Object> jobInfo = new HashMap<>();
//                JobDetail jobDetail = context.getJobDetail();
//                Trigger trigger = context.getTrigger();
//
//                jobInfo.put("jobName", jobDetail.getKey().getName());
//                jobInfo.put("jobGroup", jobDetail.getKey().getGroup());
//                jobInfo.put("triggerName", trigger.getKey().getName());
//                jobInfo.put("fireTime",
//                        LocalDateTime.ofInstant(context.getFireTime().toInstant(), ZoneId.systemDefault()));
//                jobInfo.put("scheduledFireTime",
//                        LocalDateTime.ofInstant(context.getScheduledFireTime().toInstant(), ZoneId.systemDefault()));
//                jobInfo.put("runTime", context.getJobRunTime());
//                jobInfo.put("refireCount", context.getRefireCount());
//
//                // JobData
//                JobDataMap dataMap = context.getMergedJobDataMap();
//                Map<String, Object> jobData = new HashMap<>();
//                for (String key : dataMap.getKeys()) {
//                    jobData.put(key, dataMap.get(key));
//                }
//                jobInfo.put("jobData", jobData);
//
//                result.add(jobInfo);
//            }
//
//            BaseResponse<List<Map<String, Object>>> response = BaseResponse.<List<Map<String, Object>>>builder()
//                    .status(SUCCESS)
//                    .message("Lấy danh sách jobs đang chạy thành công")
//                    .data(result)
//                    .build();
//
//            return ResponseEntity.ok(response);
//
//        } catch (Exception e) {
//            BaseResponse<List<Map<String, Object>>> response = BaseResponse.<List<Map<String, Object>>>builder()
//                    .status("ERROR")
//                    .message("Lỗi khi lấy jobs đang chạy: " + e.getMessage())
//                    .data(null)
//                    .build();
//
//            return ResponseEntity.badRequest().body(response);
//        }
//    }
//
//    @PostMapping("/jobs/{jobName}/pause")
//    public ResponseEntity<BaseResponse<String>> pauseJob(
//            @PathVariable String jobName,
//            @RequestParam(defaultValue = "DEFAULT") String jobGroup) {
//
//        try {
//            JobKey jobKey = new JobKey(jobName, jobGroup);
//
//            if (!scheduler.checkExists(jobKey)) {
//                BaseResponse<String> response = BaseResponse.<String>builder()
//                        .status("ERROR")
//                        .message("Job không tồn tại")
//                        .data(null)
//                        .build();
//                return ResponseEntity.notFound().build();
//            }
//
//            scheduler.pauseJob(jobKey);
//
//            BaseResponse<String> response = BaseResponse.<String>builder()
//                    .status(SUCCESS)
//                    .message("Đã tạm dừng job thành công")
//                    .data("Job " + jobName + " paused")
//                    .build();
//
//            return ResponseEntity.ok(response);
//
//        } catch (Exception e) {
//            BaseResponse<String> response = BaseResponse.<String>builder()
//                    .status("ERROR")
//                    .message("Lỗi khi tạm dừng job: " + e.getMessage())
//                    .data(null)
//                    .build();
//
//            return ResponseEntity.badRequest().body(response);
//        }
//    }
//
//    @PostMapping("/jobs/{jobName}/resume")
//    public ResponseEntity<BaseResponse<String>> resumeJob(
//            @PathVariable String jobName,
//            @RequestParam(defaultValue = "DEFAULT") String jobGroup) {
//
//        try {
//            JobKey jobKey = new JobKey(jobName, jobGroup);
//
//            if (!scheduler.checkExists(jobKey)) {
//                BaseResponse<String> response = BaseResponse.<String>builder()
//                        .status("ERROR")
//                        .message("Job không tồn tại")
//                        .data(null)
//                        .build();
//                return ResponseEntity.notFound().build();
//            }
//
//            scheduler.resumeJob(jobKey);
//
//            BaseResponse<String> response = BaseResponse.<String>builder()
//                    .status(SUCCESS)
//                    .message("Đã tiếp tục job thành công")
//                    .data("Job " + jobName + " resumed")
//                    .build();
//
//            return ResponseEntity.ok(response);
//
//        } catch (Exception e) {
//            BaseResponse<String> response = BaseResponse.<String>builder()
//                    .status("ERROR")
//                    .message("Lỗi khi tiếp tục job: " + e.getMessage())
//                    .data(null)
//                    .build();
//
//            return ResponseEntity.badRequest().body(response);
//        }
//    }
//
//    @DeleteMapping("/jobs/{jobName}")
//    public ResponseEntity<BaseResponse<String>> deleteJob(
//            @PathVariable String jobName,
//            @RequestParam(defaultValue = "DEFAULT") String jobGroup) {
//
//        try {
//            JobKey jobKey = new JobKey(jobName, jobGroup);
//
//            if (!scheduler.checkExists(jobKey)) {
//                BaseResponse<String> response = BaseResponse.<String>builder()
//                        .status("ERROR")
//                        .message("Job không tồn tại")
//                        .data(null)
//                        .build();
//                return ResponseEntity.notFound().build();
//            }
//
//            boolean deleted = scheduler.deleteJob(jobKey);
//
//            BaseResponse<String> response = BaseResponse.<String>builder()
//                    .status(SUCCESS)
//                    .message(deleted ? "Đã xóa job thành công" : "Không thể xóa job")
//                    .data("Job " + jobName + (deleted ? " deleted" : " not deleted"))
//                    .build();
//
//            return ResponseEntity.ok(response);
//
//        } catch (Exception e) {
//            BaseResponse<String> response = BaseResponse.<String>builder()
//                    .status("ERROR")
//                    .message("Lỗi khi xóa job: " + e.getMessage())
//                    .data(null)
//                    .build();
//
//            return ResponseEntity.badRequest().body(response);
//        }
//    }
//
//    @PostMapping("/jobs/{jobName}/trigger")
//    public ResponseEntity<BaseResponse<String>> triggerJob(
//            @PathVariable String jobName,
//            @RequestParam(defaultValue = "DEFAULT") String jobGroup) {
//
//        try {
//            JobKey jobKey = new JobKey(jobName, jobGroup);
//
//            if (!scheduler.checkExists(jobKey)) {
//                BaseResponse<String> response = BaseResponse.<String>builder()
//                        .status("ERROR")
//                        .message("Job không tồn tại")
//                        .data(null)
//                        .build();
//                return ResponseEntity.notFound().build();
//            }
//
//            scheduler.triggerJob(jobKey);
//
//            BaseResponse<String> response = BaseResponse.<String>builder()
//                    .status(SUCCESS)
//                    .message("Đã kích hoạt job thành công")
//                    .data("Job " + jobName + " triggered manually")
//                    .build();
//
//            return ResponseEntity.ok(response);
//
//        } catch (Exception e) {
//            BaseResponse<String> response = BaseResponse.<String>builder()
//                    .status("ERROR")
//                    .message("Lỗi khi kích hoạt job: " + e.getMessage())
//                    .data(null)
//                    .build();
//
//            return ResponseEntity.badRequest().body(response);
//        }
//    }
//
//    @GetMapping("/scheduler/status")
//    public ResponseEntity<BaseResponse<Map<String, Object>>> getSchedulerStatus() {
//        try {
//            Map<String, Object> status = new HashMap<>();
//
//            status.put("schedulerName", scheduler.getSchedulerName());
//            status.put("instanceId", scheduler.getSchedulerInstanceId());
//            status.put("isStarted", scheduler.isStarted());
//            status.put("isInStandby", scheduler.isInStandbyMode());
//            status.put("isShutdown", scheduler.isShutdown());
//
//            // Thêm thông tin JobStore
//            status.put("schedulerClass", scheduler.getClass().getName());
//
//            // Thống kê jobs
//            int totalJobs = 0;
//            int runningJobs = 0;
//            int pausedJobs = 0;
//
//            for (String groupName : scheduler.getJobGroupNames()) {
//                Set<JobKey> jobKeys = scheduler.getJobKeys(GroupMatcher.jobGroupEquals(groupName));
//                totalJobs += jobKeys.size();
//
//                for (JobKey jobKey : jobKeys) {
//                    List<? extends Trigger> triggers = scheduler.getTriggersOfJob(jobKey);
//                    for (Trigger trigger : triggers) {
//                        Trigger.TriggerState state = scheduler.getTriggerState(trigger.getKey());
//                        if (state == Trigger.TriggerState.NORMAL) runningJobs++;
//                        else if (state == Trigger.TriggerState.PAUSED) pausedJobs++;
//                    }
//                }
//            }
//
//            status.put("totalJobs", totalJobs);
//            status.put("runningJobs", runningJobs);
//            status.put("pausedJobs", pausedJobs);
//            status.put("currentlyExecutingJobs", scheduler.getCurrentlyExecutingJobs().size());
//
//            BaseResponse<Map<String, Object>> response = BaseResponse.<Map<String, Object>>builder()
//                    .status(SUCCESS)
//                    .message("Lấy trạng thái scheduler thành công")
//                    .data(status)
//                    .build();
//
//            return ResponseEntity.ok(response);
//
//        } catch (Exception e) {
//            BaseResponse<Map<String, Object>> response = BaseResponse.<Map<String, Object>>builder()
//                    .status("ERROR")
//                    .message("Lỗi khi lấy trạng thái scheduler: " + e.getMessage())
//                    .data(null)
//                    .build();
//
//            return ResponseEntity.badRequest().body(response);
//        }
//    }
//
//    @GetMapping("/database/jobs")
//    public ResponseEntity<BaseResponse<Map<String, Object>>> getJobsFromDatabase() {
//        try {
//            Map<String, Object> result = new HashMap<>();
//
//            // Thống kê từ database thông qua Quartz API
//            int totalJobsInDB = 0;
//            int totalTriggersInDB = 0;
//
//            List<Map<String, Object>> jobsFromDB = new ArrayList<>();
//
//            // Lấy tất cả job groups
//            List<String> jobGroups = scheduler.getJobGroupNames();
//            for (String group : jobGroups) {
//                Set<JobKey> jobKeys = scheduler.getJobKeys(GroupMatcher.jobGroupEquals(group));
//                totalJobsInDB += jobKeys.size();
//
//                for (JobKey jobKey : jobKeys) {
//                    Map<String, Object> jobInfo = new HashMap<>();
//                    jobInfo.put("jobName", jobKey.getName());
//                    jobInfo.put("jobGroup", jobKey.getGroup());
//
//                    // Kiểm tra xem job có exist không
//                    boolean exists = scheduler.checkExists(jobKey);
//                    jobInfo.put("exists", exists);
//
//                    if (exists) {
//                        JobDetail detail = scheduler.getJobDetail(jobKey);
//                        jobInfo.put("jobClass", detail.getJobClass().getSimpleName());
//                        jobInfo.put("isDurable", detail.isDurable());
//
//                        // Lấy triggers
//                        List<? extends Trigger> triggers = scheduler.getTriggersOfJob(jobKey);
//                        totalTriggersInDB += triggers.size();
//                        jobInfo.put("triggerCount", triggers.size());
//
//                        List<Map<String, Object>> triggerInfos = new ArrayList<>();
//                        for (Trigger trigger : triggers) {
//                            Map<String, Object> triggerInfo = new HashMap<>();
//                            triggerInfo.put("triggerName", trigger.getKey().getName());
//                            triggerInfo.put("state", scheduler.getTriggerState(trigger.getKey()).name());
//                            if (trigger.getNextFireTime() != null) {
//                                triggerInfo.put("nextFireTime",
//                                    LocalDateTime.ofInstant(trigger.getNextFireTime().toInstant(), ZoneId.systemDefault()));
//                            }
//                            triggerInfos.add(triggerInfo);
//                        }
//                        jobInfo.put("triggers", triggerInfos);
//                    }
//
//                    jobsFromDB.add(jobInfo);
//                }
//            }
//
//            result.put("totalJobGroups", jobGroups.size());
//            result.put("totalJobs", totalJobsInDB);
//            result.put("totalTriggers", totalTriggersInDB);
//            result.put("jobs", jobsFromDB);
//
//            BaseResponse<Map<String, Object>> response = BaseResponse.<Map<String, Object>>builder()
//                    .status(SUCCESS)
//                    .message("Lấy jobs từ database thành công")
//                    .data(result)
//                    .build();
//
//            return ResponseEntity.ok(response);
//
//        } catch (Exception e) {
//            e.printStackTrace();
//            BaseResponse<Map<String, Object>> response = BaseResponse.<Map<String, Object>>builder()
//                    .status("ERROR")
//                    .message("Lỗi khi lấy jobs từ database: " + e.getMessage())
//                    .data(null)
//                    .build();
//
//            return ResponseEntity.badRequest().body(response);
//        }
//    }
//
//    @PostMapping("/test/create-job")
//    public ResponseEntity<BaseResponse<String>> createTestJob() {
//        try {
//            // Tạo một test job để kiểm tra persistence
//            String jobName = "testJob-" + System.currentTimeMillis();
//
//            JobDetail jobDetail = JobBuilder.newJob(com.example.Auto_BE.service.MedicationReminderJob.class)
//                    .withIdentity(jobName, "TEST_GROUP")
//                    .withDescription("Test job for persistence check")
//                    .usingJobData("testData", "This is a test job")
//                    .storeDurably(true)  // Quan trọng: lưu persistent
//                    .build();
//
//            // Tạo trigger chạy sau 2 phút
//            Trigger trigger = TriggerBuilder.newTrigger()
//                    .withIdentity("testTrigger-" + System.currentTimeMillis(), "TEST_GROUP")
//                    .startAt(new Date(System.currentTimeMillis() + 120000)) // 2 phút
//                    .build();
//
//            scheduler.scheduleJob(jobDetail, trigger);
//
//            BaseResponse<String> response = BaseResponse.<String>builder()
//                    .status(SUCCESS)
//                    .message("Đã tạo test job thành công")
//                    .data("Job: " + jobName + " - Sẽ chạy sau 2 phút")
//                    .build();
//
//            return ResponseEntity.ok(response);
//
//        } catch (Exception e) {
//            BaseResponse<String> response = BaseResponse.<String>builder()
//                    .status("ERROR")
//                    .message("Lỗi khi tạo test job: " + e.getMessage())
//                    .data(null)
//                    .build();
//
//            return ResponseEntity.badRequest().body(response);
//        }
//    }
//}
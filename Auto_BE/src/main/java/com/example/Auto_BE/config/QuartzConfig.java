package com.example.Auto_BE.config;

import org.quartz.Scheduler;
import org.quartz.SchedulerException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.quartz.SchedulerFactoryBean;

import javax.sql.DataSource;

@Configuration
@EnableScheduling
public class QuartzConfig {

    @Autowired
    private DataSource dataSource;

    @Bean
    public SchedulerFactoryBean schedulerFactoryBean() {
        SchedulerFactoryBean schedulerFactory = new SchedulerFactoryBean();
        schedulerFactory.setJobFactory(new AutowiringSpringBeanJobFactory());
        schedulerFactory.setAutoStartup(true);
        
        // Set DataSource for JDBC JobStore
        schedulerFactory.setDataSource(dataSource);
        
        // Let Spring Boot handle the properties from application.yml
        schedulerFactory.setApplicationContextSchedulerContextKey("applicationContext");
        
        return schedulerFactory;
    }

    @Bean
    public Scheduler scheduler(SchedulerFactoryBean schedulerFactoryBean) throws SchedulerException {
        Scheduler scheduler = schedulerFactoryBean.getScheduler();
        if (!scheduler.isStarted()) {
            scheduler.start();
            System.out.println("Quartz Scheduler started successfully with JDBC JobStore!");
        }
        return scheduler;
    }
}

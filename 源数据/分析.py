from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import os
import glob

# 创建SparkSession
spark = SparkSession.builder \
    .appName("ShenzhenTransportAnalysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# 设置日志级别
spark.sparkContext.setLogLevel("WARN")

print("=== Spark Session 初始化完成 ===")
print(f"Spark版本: {spark.version}")

# 定义数据schema
schema = StructType([
    StructField("card_id", StringType(), True),
    StructField("trans_time", StringType(), True),
    StructField("trans_type", StringType(), True),
    StructField("trans_amount", IntegerType(), True),
    StructField("balance", IntegerType(), True),
    StructField("station_line_code", StringType(), True),
    StructField("line_name", StringType(), True),
    StructField("station_name", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("status", IntegerType(), True),
    StructField("settle_time", StringType(), True)
])

# 读取CSV数据
try:
    df = spark.read \
        .option("header", "false") \
        .schema(schema) \
        .csv("SZTcard.csv")

    print("数据读取成功!")
    print(f"总记录数: {df.count()}")
except Exception as e:
    print(f"数据读取失败: {e}")
    exit(1)

# 数据预处理：转换时间格式
df_clean = df.withColumn("trans_time", to_timestamp(col("trans_time"), "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("settle_time", to_timestamp(col("settle_time"), "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("trans_date", to_date(col("trans_time"))) \
    .withColumn("trans_hour", hour(col("trans_time")))

print("数据概览：")
df_clean.show(10)
print(f"总记录数: {df_clean.count()}")

# 创建输出目录
output_dir = "/tmp/transport_analysis_csv"
os.makedirs(output_dir, exist_ok=True)
print(f"分析结果将保存到: {output_dir}")


# 保存函数 - 将结果保存为单个CSV文件
def save_as_single_csv(df, table_name, description=""):
    """将分析结果保存为单个CSV文件"""
    try:
        if df.count() > 0:
            # 临时目录
            temp_dir = f"{output_dir}/temp_{table_name}"

            # 保存为CSV文件
            df.coalesce(1).write \
                .mode("overwrite") \
                .option("header", "true") \
                .csv(temp_dir)

            # 查找生成的CSV文件并重命名为单个文件
            csv_files = glob.glob(f"{temp_dir}/part-*.csv")
            if csv_files:
                # 取第一个CSV文件
                source_file = csv_files[0]
                # 目标文件名
                target_file = f"{output_dir}/{table_name}.csv"

                # 重命名文件
                os.rename(source_file, target_file)

                # 删除临时目录
                import shutil
                shutil.rmtree(temp_dir)

                print(f"✓ 成功保存: {table_name}.csv - {description}")
                print(f"  文件位置: {target_file}")
                return True
            else:
                print(f"✗ 未找到CSV文件: {table_name}")
                return False
        else:
            print(f"⚠ 数据为空，跳过保存: {table_name}")
            return False
    except Exception as e:
        print(f"✗ 保存失败 {table_name}: {e}")
        return False


# 1. 基础统计分析
print("\n" + "=" * 50)
print("1. 基础统计分析")
print("=" * 50)

# 交易类型分布
print("交易类型分布：")
trans_type_stats = df_clean.groupBy("trans_type").agg(
    count("*").alias("count"),
    format_number(count("*") / df_clean.count() * 100, 2).alias("percentage")
).orderBy(desc("count"))
trans_type_stats.show()

save_as_single_csv(trans_type_stats, "trans_type_stats", "交易类型分布统计")

# 线路使用情况
print("线路使用情况（前20名）：")
line_stats = df_clean.filter(col("line_name").isNotNull()) \
    .groupBy("line_name").agg(count("*").alias("count")) \
    .orderBy(desc("count"))
line_stats.show(20, truncate=False)

save_as_single_csv(line_stats, "line_usage_stats", "线路使用情况统计")

# 2. 地铁出行分析
print("\n" + "=" * 50)
print("2. 地铁出行分析")
print("=" * 50)

# 分离地铁入站和出站记录
metro_enter = df_clean.filter((col("trans_type") == "地铁入站") & (col("station_name").isNotNull()))
metro_exit = df_clean.filter((col("trans_type") == "地铁出站") & (col("station_name").isNotNull()))

print(f"地铁入站记录数: {metro_enter.count()}")
print(f"地铁出站记录数: {metro_exit.count()}")

# 热门地铁站点
print("热门入站站点（前15名）：")
top_enter_stations = metro_enter.groupBy("station_name").agg(
    count("*").alias("enter_count")
).orderBy(desc("enter_count"))
top_enter_stations.show(15, truncate=False)

save_as_single_csv(top_enter_stations, "top_enter_stations", "热门入站站点")

print("热门出站站点（前15名）：")
top_exit_stations = metro_exit.groupBy("station_name").agg(
    count("*").alias("exit_count")
).orderBy(desc("exit_count"))
top_exit_stations.show(15, truncate=False)

save_as_single_csv(top_exit_stations, "top_exit_stations", "热门出站站点")

# 3. 时间分析
print("\n" + "=" * 50)
print("3. 时间分析")
print("=" * 50)

# 小时分布
print("交易小时分布：")
hourly_stats = df_clean.groupBy("trans_hour").agg(
    count("*").alias("count")
).orderBy("trans_hour")
hourly_stats.show(24)

save_as_single_csv(hourly_stats, "hourly_transaction_stats", "小时交易分布")

# 4. 巴士分析
print("\n" + "=" * 50)
print("4. 巴士分析")
print("=" * 50)

bus_data = df_clean.filter(col("trans_type") == "巴士")

print("巴士线路使用情况：")
bus_line_stats = bus_data.groupBy("line_name").agg(
    count("*").alias("count"),
    avg("trans_amount").alias("avg_amount"),
    sum("trans_amount").alias("total_amount")
).orderBy(desc("count"))
bus_line_stats.show(truncate=False)

save_as_single_csv(bus_line_stats, "bus_line_stats", "巴士线路统计")

# 5. 用户行为分析
print("\n" + "=" * 50)
print("5. 用户行为分析")
print("=" * 50)

# 计算每张卡的交易次数
user_activity = df_clean.groupBy("card_id").agg(
    count("*").alias("trans_count"),
    countDistinct("trans_date").alias("active_days"),
    min("trans_time").alias("first_trans"),
    max("trans_time").alias("last_trans")
).filter(col("trans_count") > 1)

print("用户活动统计：")
user_activity.select(
    avg("trans_count").alias("avg_trans_per_user"),
    avg("active_days").alias("avg_active_days"),
    count("*").alias("total_users")
).show()

save_as_single_csv(user_activity, "user_activity", "用户活动详情")

# 高频用户（前100名）
print("高频用户（前100名）：")
top_users = user_activity.orderBy(desc("trans_count")).limit(100)
top_users.show(20, truncate=False)

save_as_single_csv(top_users, "top_users", "高频用户TOP100")

# 6. 消费分析
print("\n" + "=" * 50)
print("6. 消费分析")
print("=" * 50)

# 地铁票价分析
metro_fare_stats = metro_exit.agg(
    count("*").alias("total_records"),
    avg("trans_amount").alias("avg_fare"),
    min("trans_amount").alias("min_fare"),
    max("trans_amount").alias("max_fare"),
    sum("trans_amount").alias("total_revenue")
)
print("地铁票价统计：")
metro_fare_stats.show()

save_as_single_csv(metro_fare_stats, "metro_fare_stats", "地铁票价统计")

# 巴士票价分析
bus_fare_stats = bus_data.agg(
    count("*").alias("total_records"),
    avg("trans_amount").alias("avg_fare"),
    min("trans_amount").alias("min_fare"),
    max("trans_amount").alias("max_fare"),
    sum("trans_amount").alias("total_revenue")
)
print("巴士票价统计：")
bus_fare_stats.show()

save_as_single_csv(bus_fare_stats, "bus_fare_stats", "巴士票价统计")

# 7. 设备使用分析
print("\n" + "=" * 50)
print("7. 设备使用分析")
print("=" * 50)

print("热门地铁设备（前20名）：")
device_stats = df_clean.filter(col("device_id").isNotNull()) \
    .groupBy("device_id").agg(
    count("*").alias("usage_count"),
    countDistinct("card_id").alias("unique_users")
).orderBy(desc("usage_count"))
device_stats.show(20, truncate=False)

save_as_single_csv(device_stats, "device_usage_stats", "设备使用统计")

# 8. 生成分析报告
print("\n" + "=" * 50)
print("8. 分析报告摘要")
print("=" * 50)

# 创建临时视图用于SQL查询
df_clean.createOrReplaceTempView("transport_data")

# 使用SQL进行复杂查询
summary_report = spark.sql("""
SELECT 
    trans_date,
    COUNT(*) as daily_transactions,
    COUNT(DISTINCT card_id) as daily_users,
    SUM(CASE WHEN trans_type = '地铁出站' THEN trans_amount ELSE 0 END) as metro_revenue,
    SUM(CASE WHEN trans_type = '巴士' THEN trans_amount ELSE 0 END) as bus_revenue
FROM transport_data
GROUP BY trans_date
ORDER BY trans_date
""")

print("每日交易汇总：")
summary_report.show()

save_as_single_csv(summary_report, "daily_summary", "每日交易汇总")

# 显示所有生成的CSV文件
print("\n" + "=" * 50)
print("生成的CSV文件汇总")
print("=" * 50)

print(f"输出目录: {output_dir}")
print("\n生成的CSV文件:")
csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
for csv_file in sorted(csv_files):
    file_path = os.path.join(output_dir, csv_file)
    file_size = os.path.getsize(file_path)
    print(f"  {csv_file} ({file_size} bytes)")

# 生成Hive导入指南
print("\n" + "=" * 50)
print("Hive导入指南")
print("=" * 50)

print("""
将CSV文件导入Hive的步骤:

1. 将CSV文件上传到HDFS:
   hdfs dfs -put /tmp/transport_analysis_csv/*.csv /user/hive/csv_data/

2. 在Hive中创建表 (以trans_type_stats为例):
   CREATE TABLE xingye.trans_type_stats (
     trans_type STRING,
     count BIGINT,
     percentage STRING
   ) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE;

3. 加载数据到Hive表:
   LOAD DATA INPATH '/user/hive/csv_data/trans_type_stats.csv' INTO TABLE xingye.trans_type_stats;

4. 验证数据:
   SELECT * FROM xingye.trans_type_stats LIMIT 10;

重复步骤2-4为每个CSV文件创建对应的Hive表。
""")

print("\n" + "=" * 50)
print("分析完成!")
print("=" * 50)
print(f"所有分析结果已保存为单个CSV文件到: {output_dir}")
print(f"共生成 {len(csv_files)} 个CSV文件")

# 停止SparkSession
spark.stop()
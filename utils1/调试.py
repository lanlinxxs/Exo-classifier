import os
import sys

# 设置R路径
r_path = r"D:\R\R-4.3.3"
os.environ['R_HOME'] = r_path
os.environ['PATH'] = f"{r_path}\\bin\\x64;{os.environ['PATH']}"

# 验证路径
print(f"R_HOME设置为: {os.environ['R_HOME']}")
print(f"PATH中包含: {[p for p in os.environ['PATH'].split(';') if 'R' in p]}")

try:
    from rpy2.robjects.packages import importr
    base = importr('base')
    print(f"R.home()返回: {base.r['R.home']()[0]}")
    print("✅ R环境配置成功！")
except Exception as e:
    print(f"❌ 加载失败: {str(e)}")
    print("\n调试建议：")
    print(f"1. 确认路径 {r_path} 存在且包含bin/x64/R.dll")
    print("2. 以管理员身份运行此脚本")
    print("3. 检查R是否为64位（需与Python位数匹配）")
    sys.exit(1)
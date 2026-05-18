import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RelayController:
    def __init__(self, exe_path: str, serial_number: str):
        self.exe_path = exe_path
        self.serial_number = serial_number

    def _execute_command(self, action: str, channel: str) -> bool:
        try:
            command = [self.exe_path, self.serial_number, action, channel]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"成功执行命令: {' '.join(command)}")
                return True
            else:
                logger.error(f"命令执行失败: {' '.join(command)}")
                logger.error(f"返回码: {result.returncode}")
                logger.error(f"错误输出: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"执行命令时发生异常: {' '.join(command)}")
            logger.error(f"异常信息: {str(e)}")
            return False

    def turn_on(self, channel: int) -> bool:
        if not (1 <= channel <= 8):
            logger.error(f"无效的通道号: {channel}，必须在 1-8 之间")
            return False

        channel_str = f"{channel:02d}"
        return self._execute_command('open', channel_str)

    def turn_off(self, channel: int) -> bool:
        if not (1 <= channel <= 8):
            logger.error(f"无效的通道号: {channel}，必须在 1-8 之间")
            return False

        channel_str = f"{channel:02d}"
        return self._execute_command('close', channel_str)

    def turn_all_on(self) -> bool:
        return self._execute_command('open', '255')

    def turn_all_off(self) -> bool:
        return self._execute_command('close', '255')


def show_menu():
    print("\n===== USB 继电器控制器 =====")
    print("1. 打开指定通道继电器")
    print("2. 关闭指定通道继电器")
    print("3. 打开所有继电器")
    print("4. 关闭所有继电器")
    print("5. 退出")
    print("=============================")


if __name__ == "__main__":
    relay = RelayController(
        exe_path=r"d:\BaiduNetdiskDownload\4roadcontrol\TestApp\CommandApp_USBRelay.exe",
        serial_number="HW341"
    )
    
    print("USB 继电器控制器已启动")
    
    while True:
        show_menu()
        choice = input("请输入操作编号 (1-5): ")
        
        if choice == "1":
            try:
                channel = int(input("请输入通道号 (1-8): "))
                relay.turn_on(channel)
            except ValueError:
                print("请输入有效的数字")
        elif choice == "2":
            try:
                channel = int(input("请输入通道号 (1-8): "))
                relay.turn_off(channel)
            except ValueError:
                print("请输入有效的数字")
        elif choice == "3":
            relay.turn_all_on()
        elif choice == "4":
            relay.turn_all_off()
        elif choice == "5":
            print("退出程序...")
            break
        else:
            print("无效的选项，请输入 1-5")
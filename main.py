"""BLACKLEDGER — Application entry point (production-ready)."""
from logger import get_logger
from views.main_window import MainWindow
from controllers.app_controller import AppController

def main():
    log = get_logger("blackledger.main")
    log.info("Starting BLACKLEDGER …")

    # 1. Create the view with a placeholder controller
    view = MainWindow(controller=None)

    # 2. Create the real controller and wire it in
    controller = AppController(view)
    view.controller = controller

    log.info("BLACKLEDGER ready.")
    view.run()

if __name__ == "__main__":
    main()
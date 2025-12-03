import asyncio
from mavsdk import System


async def run():
    # 1) Connect to PX4 SITL
    drone = System()
    print("Connecting to PX4 SITL...")
    await drone.connect(system_address="udp://:14540")

    # 2) Wait until connected
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Connected to PX4!")
            break

    # 3) Wait for basic health
    print("Waiting for drone to be ready (health checks)...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("✅ Drone health OK, ready to arm")
            break
        print("  Still waiting for health OK...")
        await asyncio.sleep(1)

    # 4) Arm
    print("Arming drone...")
    await drone.action.arm()
    print("✅ Armed")

    # 5) Takeoff
    print("Taking off...")
    await drone.action.set_takeoff_altitude(5.0)
    await drone.action.takeoff()

    await asyncio.sleep(8)
    print("Drone should now be hovering around 5 meters.")

    # 6) Hover
    print("Hovering for 5 seconds...")
    await asyncio.sleep(5)

    # 7) Land
    print("Landing...")
    await drone.action.land()

    # Wait until landed
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("✅ Landed and disarmed")
            break
        await asyncio.sleep(1)

    print("Demo complete. Closing connection.")


if __name__ == "__main__":
    asyncio.run(run())

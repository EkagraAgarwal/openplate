"""python -m platesim.mcp3d --root <runs_dir>"""
import argparse

from .server import create_server


def main():
    parser = argparse.ArgumentParser(description="3-D plating MCP server (stdio)")
    parser.add_argument("--root", required=True, help="Directory containing isolated runs")
    args = parser.parse_args()
    create_server(args.root).run(transport="stdio")


if __name__ == "__main__":
    main()

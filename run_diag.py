from src.main import show_side_by_side
from src.maps import example_map_obstacles

if __name__=='__main__':
    grid = example_map_obstacles()
    start=(0,0); goal=(7,7)
    fig, r1, r2 = show_side_by_side(grid,start,goal)
    print('done', r1[2], r2[2])

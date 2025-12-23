import tkinter as tk

from applications import graph


class Maze:

    def __init__(self):
        self._window = tk.Tk()
        self._window.geometry("400x500")
        self._window.title("Shortest Path Finder")
        self._canvas = tk.Canvas(self._window, width=400, height=400)
        self._canvas.pack()
        self.G = graph.AdjMatrixUndiGraph(100)
        self.bfs = graph.BFS(self.G)
        self._is_left_clicked = False

        self._start: tuple = ()
        self._end: tuple = ()

        self.path: list = []

        self._btn_barrier = tk.Button(self._window, text='Add Barriers', command=self._onclick_btn_barrier)
        self._btn_barrier.pack(side='top')
        self._btn_barrier_done = tk.Button(self._window, text='Done', command=self._onclick_btn_barrier_done)
        self._btn_find_path = tk.Button(self._window, text='Find Path', command=self._onclick_btn_find_path)
        self._btn_reset = tk.Button(self._window, text='Reset', command=self._onclick_btn_reset)

        self._in_barrier_state = False
        self._in_vertex_choosing_state = False
        self._chose_starting_point = False
        self._chose_end_point = False

        self._lbl_starting_point = tk.Label(self._window, text='Choose a starting point')
        self._lbl_ending_point = tk.Label(self._window, text='Choose a destination point')

    def execute(self) -> None:
        """
        Executes the main loop.
        :return: None
        """
        self._draw_pixels()

        self._canvas.bind('<Button-1>', self._left_click)
        self._canvas.bind('<B1-Motion>', self._drag)

        self._window.mainloop()

    def _onclick_btn_barrier(self) -> None:
        """
        Defines the action on click of the Barrier button
        :return: None
        """
        self._in_barrier_state = True
        self._btn_barrier.pack_forget()
        self._btn_barrier_done.pack(side='top')

    def _onclick_btn_barrier_done(self) -> None:
        """
        Defines the action on click of the Done button
        :return: None
        """
        self._in_barrier_state = False
        self._btn_barrier_done.pack_forget()
        self._in_vertex_choosing_state = True
        self._lbl_starting_point.pack(side='top')

    def _onclick_btn_find_path(self) -> None:
        """
        Defines the action on click of the find path button.
        :return: None
        """
        self.path = self.bfs.path_to(self._index_to_vertex(self._start[0], self._start[1]),
                                     self._index_to_vertex(self._end[0], self._end[1]))
        self._fill_path()
        self._btn_find_path.pack_forget()
        self._btn_reset.pack(side='top')

    def _onclick_btn_reset(self) -> None:
        """
        Clears the canvas and resets the state of the maze.
        :return: None
        """
        self._btn_reset.pack_forget()
        self._canvas.delete("all")
        self._draw_pixels()
        self._start = ()
        self._end = ()
        self.path = []
        self._in_barrier_state = False
        self._in_vertex_choosing_state = False
        self._chose_starting_point = False
        self._chose_end_point = False
        self._btn_barrier.pack(side='top')
        self._btn_find_path.pack_forget()

    def _left_click(self, event: tk.Event) -> None:
        """
        Defines what the left click action will perform. In this case it's used to determine
        the starting and the destination points.
        :param event: Event
        :return: None
        """
        if self._is_outside_grid(event.x, event.y):
            return
        if self._in_vertex_choosing_state:
            if not self._chose_starting_point:
                self._chose_starting_point = True
                self._start = int(event.y / 40), int(event.x / 40)
                self._fill_vertex(self._start[0], self._start[1], '#2ecc71')
                self._lbl_starting_point.pack_forget()
                self._lbl_ending_point.pack(side='top')
                print(f'Clicked vertex coordinates: {self._start}')
            elif not self._chose_end_point:
                self._chose_end_point = True
                self._end = int(event.y / 40), int(event.x / 40)
                self._fill_vertex(self._end[0], self._end[1], '#e74c3c')
                print(f'Clicked vertex coordinates: {self._end}')
                self._lbl_ending_point.pack_forget()
                self._btn_find_path.pack()

    def _drag(self, event: tk.Event) -> None:
        """
        Defines what the drag action will perform. In this case it's used to draw barriers.
        :param event: Event
        :return: None
        """
        if self._is_outside_grid(event.x, event.y):
            return
        if self._in_barrier_state:
            i, j = int(event.y / 40), int(event.x / 40)
            self.G.clear_edges_of_vertex(self._index_to_vertex(i, j))
            self._fill_vertex(i, j, '#34495e')

    def _fill_vertex(self, i, j, color='red') -> None:
        """
        Fills the given vertex indices with the specified color
        :param i: i coordinate of the vertex
        :param j: j coordinate of the vertex
        :param color: color
        :return: None
        """
        x0 = j * 40
        y0 = i * 40
        x1 = x0 + 40
        y1 = y0 + 40
        self._canvas.create_rectangle(x0, y0, x1, y1, fill=color)

    def _fill_path(self) -> None:
        """
        Fills the self.path
        :return: None
        """
        for vertex in self.path:
            i, j = self._vertex_to_index(vertex)
            self._fill_vertex(i, j, '#3498db')

    def _draw_pixels(self) -> None:
        """
        Draws the pixel grid 10x10
        :return: None
        """
        for i in range(10):
            for j in range(10):
                x0 = 40 * j
                x1 = x0 + 40
                y0 = 40 * i
                y1 = y0 + 40
                self._canvas.create_rectangle(x0, y0, x1, y1)
                self._connect_pixels(i, j)

    def _connect_pixels(self, i: int, j: int) -> None:
        """
        Makes connection between the given vertex index with around vertices in the Graph representation.
        :param i: i coordinate of the vertex
        :param j: j coordinate of the vertex
        :return: None
        """
        curr_vertex = self._index_to_vertex(i, j)

        # Connect with the left vertex
        if i - 1 >= 0:
            left_vertex = self._index_to_vertex(i - 1, j)
            self.G.add_edge(curr_vertex, left_vertex)

        # Connect with the right vertex
        if i + 1 <= 9:
            right_vertex = self._index_to_vertex(i + 1, j)
            self.G.add_edge(curr_vertex, right_vertex)

        # Connect with the above vertex
        if j - 1 >= 0:
            above_vertex = self._index_to_vertex(i, j - 1)
            self.G.add_edge(curr_vertex, above_vertex)

        # Connect with the below vertex
        if j + 1 <= 9:
            below_vertex = self._index_to_vertex(i, j + 1)
            self.G.add_edge(curr_vertex, below_vertex)

    def _vertex_to_index(self, vertex: int) -> tuple:
        """
        Converts the vertex id to indices
        :param vertex: vertex id
        :return: a tuple if indices (i,j)
        """
        str_vertex = str(vertex)
        if len(str_vertex) > 1:
            return int(str_vertex[0]), int(str_vertex[1])

        return 0, int(str_vertex[0])

    def _index_to_vertex(self, i: int, j: int) -> int:
        """
        Converts indices to vertex id.
        :param i: i coordinate
        :param j: j coordinate
        :return: vertex id
        """
        return int(str(i) + str(j))

    def _is_outside_grid(self, x, y) -> bool:
        """
        Checks whether the coordinates are inside the grid or not.
        :param x: x coordinate
        :param y: y coordinate
        :return: True if the coordinates are inside the grid, False otherwise
        """
        return x < 0 or x > 400 or y < 0 or y > 400


m = Maze()
m.execute()
print(m.path)
graph.visualize(m.G)

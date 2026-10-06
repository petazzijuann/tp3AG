"""
=============================================================
  TRABAJO PRÁCTICO N°3 - Problema del Viajante (TSP)
  23 capitales de provincia de la República Argentina
  Opciones:
    1) Análisis del método exhaustivo (ejercicio 1)
    a) Heurística "vecino más cercano" desde una capital elegida
    b) Mejor recorrido con la heurística (probando todas las capitales)
    c) Algoritmo genético (crossover cíclico)
=============================================================
"""

import json          # Para leer el contorno de las provincias (mapa)
import math          # Para la distancia haversine y factoriales
import os            # Para armar rutas de archivos
import random        # Para generar números aleatorios
import time          # Para medir tiempo de ejecución
import itertools     # Para generar permutaciones (método exhaustivo)

import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  PARÁMETROS DEL ALGORITMO GENÉTICO
# ─────────────────────────────────────────────

N_POB          = 50     # N: cantidad de cromosomas de la población
M_CICLOS       = 200    # M: cantidad de ciclos (generaciones)
PC             = 0.90   # Probabilidad de crossover (cíclico)
PM             = 0.80   # Probabilidad de mutación por individuo (inversión de un tramo)
TAM_TORNEO     = 3      # Individuos que compiten en cada torneo
ELITISMO       = 2      # Mejores individuos que pasan directo a la siguiente generación

CARPETA        = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_MAPA   = os.path.join(CARPETA, "argentina_provincias.json")

# ─────────────────────────────────────────────
#  DATOS: CAPITALES DE PROVINCIA (lat, lon)
# ─────────────────────────────────────────────
# Cada gen del cromosoma es el número de ciudad (1 a 23) de esta tabla.

CAPITALES = [
    # (provincia,             capital,                              latitud,  longitud)
    ("Buenos Aires",          "La Plata",                           -34.921, -57.955),
    ("Catamarca",             "San Fernando del Valle de Catamarca", -28.469, -65.779),
    ("Chaco",                 "Resistencia",                        -27.451, -58.987),
    ("Chubut",                "Rawson",                             -43.300, -65.102),
    ("Córdoba",               "Córdoba",                            -31.420, -64.188),
    ("Corrientes",            "Corrientes",                         -27.469, -58.830),
    ("Entre Ríos",            "Paraná",                             -31.732, -60.529),
    ("Formosa",               "Formosa",                            -26.185, -58.173),
    ("Jujuy",                 "San Salvador de Jujuy",              -24.186, -65.299),
    ("La Pampa",              "Santa Rosa",                         -36.620, -64.290),
    ("La Rioja",              "La Rioja",                           -29.413, -66.856),
    ("Mendoza",               "Mendoza",                            -32.889, -68.845),
    ("Misiones",              "Posadas",                            -27.367, -55.896),
    ("Neuquén",               "Neuquén",                            -38.952, -68.059),
    ("Río Negro",             "Viedma",                             -40.813, -62.997),
    ("Salta",                 "Salta",                              -24.782, -65.423),
    ("San Juan",              "San Juan",                           -31.537, -68.536),
    ("San Luis",              "San Luis",                           -33.301, -66.338),
    ("Santa Cruz",            "Río Gallegos",                       -51.623, -69.216),
    ("Santa Fe",              "Santa Fe",                           -31.633, -60.700),
    ("Santiago del Estero",   "Santiago del Estero",                -27.795, -64.261),
    ("Tierra del Fuego",      "Ushuaia",                            -54.801, -68.303),
    ("Tucumán",               "San Miguel de Tucumán",              -26.808, -65.218),
]

N_CIUDADES = len(CAPITALES)   # 23


# ─────────────────────────────────────────────
#  MATRIZ DE DISTANCIAS D (N x N)
# ─────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    """
    Distancia en km entre dos puntos de la Tierra (distancia ortodrómica,
    es decir, en línea recta sobre la superficie esférica).
    """
    R = 6371.0  # Radio medio de la Tierra en km
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def construir_matriz() -> list:
    """
    Arma la matriz D donde D[x][y] es la distancia (km) entre la ciudad x y la ciudad y.
    Es simétrica (D[x][y] == D[y][x]) y con diagonal 0.
    """
    D = [[0.0] * N_CIUDADES for _ in range(N_CIUDADES)]
    for i in range(N_CIUDADES):
        for j in range(i + 1, N_CIUDADES):
            d = haversine(CAPITALES[i][2], CAPITALES[i][3], CAPITALES[j][2], CAPITALES[j][3])
            D[i][j] = D[j][i] = d
    return D


D = construir_matriz()


def longitud_ruta(ruta: list) -> float:
    """
    Longitud total del recorrido CERRADO: suma las distancias entre ciudades
    consecutivas y además la vuelta desde la última ciudad a la de partida.
    """
    total = 0.0
    for k in range(len(ruta)):
        total += D[ruta[k]][ruta[(k + 1) % len(ruta)]]
    return total


def rotar_desde(ruta: list, inicio: int) -> list:
    """
    Un recorrido cerrado es el mismo empiece donde empiece; esta función lo
    rota para que arranque en la ciudad 'inicio' (solo para mostrarlo).
    """
    k = ruta.index(inicio)
    return ruta[k:] + ruta[:k]


# ─────────────────────────────────────────────
#  SALIDA POR PANTALLA Y MAPA
# ─────────────────────────────────────────────

def nombre(i: int) -> str:
    return f"{CAPITALES[i][1]} ({CAPITALES[i][0]})"


def mostrar_ruta(ruta: list, titulo: str):
    """Imprime ciudad de partida, recorrido completo (con regreso) y longitud."""
    print("\n" + "=" * 62)
    print(f"  {titulo}")
    print("=" * 62)
    print(f"  Ciudad de partida: {nombre(ruta[0])}")
    print("  Recorrido:")
    for paso, c in enumerate(ruta + [ruta[0]]):
        tramo = "" if paso == 0 else f"   (+{D[ruta[paso - 1]][c]:7.1f} km)"
        print(f"    {paso:2d}. {nombre(c)}{tramo}")
    print(f"  Longitud total del trayecto: {longitud_ruta(ruta):,.1f} km")
    print("=" * 62)


def dibujar_mapa(ruta: list, titulo: str, archivo: str = None):
    """
    Dibuja el mapa de la República Argentina (contorno de provincias) con el
    recorrido indicado. La ciudad de partida se marca con una estrella roja.
    """
    fig, ax = plt.subplots(figsize=(7, 11))

    # Contorno de provincias
    if os.path.exists(ARCHIVO_MAPA):
        with open(ARCHIVO_MAPA, encoding="utf-8") as f:
            provincias = json.load(f)
        for prov in provincias:
            for anillo in prov["anillos"]:
                xs = [p[0] for p in anillo]
                ys = [p[1] for p in anillo]
                ax.fill(xs, ys, color="#eef2f7", edgecolor="#9aa7b8", linewidth=0.6, zorder=1)

    # Recorrido (cerrado)
    cerrado = ruta + [ruta[0]]
    xs = [CAPITALES[c][3] for c in cerrado]
    ys = [CAPITALES[c][2] for c in cerrado]
    ax.plot(xs, ys, "-", color="#2E4A7D", linewidth=1.8, zorder=2)

    # Flechas para indicar el sentido del viaje
    for k in range(len(cerrado) - 1):
        ax.annotate("", xy=(xs[k + 1], ys[k + 1]), xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="-|>", color="#2E4A7D", lw=0, mutation_scale=11,
                                    shrinkA=0, shrinkB=6), zorder=3)

    # Ciudades numeradas según el orden de visita
    ubicadas = []
    for orden, c in enumerate(ruta):
        lat, lon = CAPITALES[c][2], CAPITALES[c][3]
        ax.plot(lon, lat, "o", color="#2E4A7D", markersize=5, zorder=4)
        # Si ya hay una ciudad muy cerca (ej: Santa Fe y Paraná), la etiqueta va abajo
        cerca = any(abs(lat - a) < 0.8 and abs(lon - b) < 0.8 for a, b in ubicadas)
        ax.annotate(f"{orden + 1}. {CAPITALES[c][1]}", (lon, lat),
                    xytext=(4, -10) if cerca else (4, 3),
                    textcoords="offset points", fontsize=7, zorder=5)
        ubicadas.append((lat, lon))
    lat0, lon0 = CAPITALES[ruta[0]][2], CAPITALES[ruta[0]][3]
    ax.plot(lon0, lat0, "*", color="#d62728", markersize=16, zorder=6, label="Partida")

    ax.set_title(f"{titulo}\nLongitud: {longitud_ruta(ruta):,.1f} km", fontsize=11)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect(1 / math.cos(math.radians(38)))  # Corrige la deformación por latitud
    ax.set_xlim(-74.5, -52.5)
    ax.set_ylim(-56, -21)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if archivo:
        fig.savefig(os.path.join(CARPETA, archivo), dpi=150)
        print(f"  Mapa guardado en: {archivo}")
    plt.show()


# ─────────────────────────────────────────────
#  EJERCICIO 1: MÉTODO EXHAUSTIVO
# ─────────────────────────────────────────────

def exhaustivo(ciudades: list) -> tuple:
    """
    Prueba TODOS los recorridos posibles de las ciudades dadas.
    Se fija la primera ciudad (el recorrido es cerrado, así que da igual
    dónde empiece) y se permutan las demás: (n-1)! recorridos.
    """
    inicio, resto = ciudades[0], ciudades[1:]
    mejor, mejor_d = None, float("inf")
    for perm in itertools.permutations(resto):
        ruta = [inicio] + list(perm)
        d = longitud_ruta(ruta)
        if d < mejor_d:
            mejor, mejor_d = ruta, d
    return mejor, mejor_d


def analisis_exhaustivo():
    """
    Resuelve el TSP por fuerza bruta para pocas ciudades, mide cuánto tarda
    cada recorrido y extrapola cuánto tardaría con las 23 capitales.
    """
    print("\n  Método exhaustivo: se resuelve para n = 6..10 ciudades y se extrapola.\n")
    print(f"  {'n':>3} | {'recorridos (n-1)!':>18} | {'tiempo (s)':>10} | {'s / recorrido':>13}")
    print("  " + "-" * 55)
    seg_por_ruta = 0
    for n in range(6, 11):
        t0 = time.perf_counter()
        _, d = exhaustivo(list(range(n)))
        t = time.perf_counter() - t0
        rutas = math.factorial(n - 1)
        seg_por_ruta = t / rutas
        print(f"  {n:>3} | {rutas:>18,} | {t:>10.3f} | {seg_por_ruta:>13.2e}")

    rutas_23 = math.factorial(N_CIUDADES - 1)
    distintas = rutas_23 // 2   # Cada recorrido se cuenta dos veces (ida y vuelta)
    anios_py = rutas_23 * seg_por_ruta / (3600 * 24 * 365)
    anios_rapida = distintas / 1e9 / (3600 * 24 * 365)
    print(f"\n  Con las {N_CIUDADES} capitales:")
    print(f"    Recorridos a evaluar  (n-1)!   = {rutas_23:,}  (~{rutas_23:.2e})")
    print(f"    Recorridos distintos  (n-1)!/2 = {distintas:,}  (~{distintas:.2e})")
    print(f"    Tiempo estimado con este programa: ~{anios_py:.2e} años")
    print(f"    Aun evaluando 1.000 millones de recorridos por segundo: ~{anios_rapida:,.0f} años")
    print("  => No es posible resolverlo por búsqueda exhaustiva en un tiempo razonable.")


# ─────────────────────────────────────────────
#  EJERCICIO 2a / 2b: HEURÍSTICA VECINO MÁS CERCANO
# ─────────────────────────────────────────────

def vecino_mas_cercano(inicio: int) -> list:
    """
    "Desde cada ciudad ir a la ciudad más cercana no visitada."
    Al terminar, el regreso a la ciudad de partida está incluido en longitud_ruta().
    """
    ruta = [inicio]
    no_visitadas = set(range(N_CIUDADES)) - {inicio}
    while no_visitadas:
        actual = ruta[-1]
        siguiente = min(no_visitadas, key=lambda c: D[actual][c])
        ruta.append(siguiente)
        no_visitadas.remove(siguiente)
    return ruta


def elegir_capital() -> int:
    print("\n  Capitales disponibles:")
    for i, (prov, cap, _, _) in enumerate(CAPITALES):
        print(f"    {i + 1:2d}. {prov:<20} - {cap}")
    while True:
        op = input("\n  Ingrese el número de la provincia de partida (1-23): ").strip()
        if op.isdigit() and 1 <= int(op) <= N_CIUDADES:
            return int(op) - 1
        print("  Opción inválida.")


def opcion_a():
    inicio = elegir_capital()
    ruta = vecino_mas_cercano(inicio)
    mostrar_ruta(ruta, f"Heurística vecino más cercano - partida: {CAPITALES[inicio][1]}")
    dibujar_mapa(ruta, f"Vecino más cercano desde {CAPITALES[inicio][1]}",
                 f"mapa_heuristica_{inicio + 1}.png")


def mejor_heuristica() -> tuple:
    """Aplica la heurística partiendo de cada una de las 23 capitales y se queda con la mejor."""
    resultados = []
    for i in range(N_CIUDADES):
        ruta = vecino_mas_cercano(i)
        resultados.append((longitud_ruta(ruta), i, ruta))
    resultados.sort()
    return resultados


def opcion_b():
    resultados = mejor_heuristica()
    print("\n  Longitud obtenida según la capital de partida:")
    for d, i, _ in resultados:
        print(f"    {CAPITALES[i][1]:<38} {d:>10,.1f} km")
    d, i, ruta = resultados[0]
    mostrar_ruta(ruta, "Recorrido mínimo con la heurística (mejor de las 23 partidas)")
    dibujar_mapa(ruta, f"Mejor heurística (partida: {CAPITALES[i][1]})", "mapa_mejor_heuristica.png")


# ─────────────────────────────────────────────
#  EJERCICIO 2c: ALGORITMO GENÉTICO
# ─────────────────────────────────────────────
# Cromosoma: permutación de las 23 ciudades (cada gen es una ciudad).
# Función objetivo: longitud del recorrido cerrado (se busca MINIMIZAR).
# Fitness: 1 / longitud (a menor distancia, mayor aptitud).

def crear_cromosoma() -> list:
    cromosoma = list(range(N_CIUDADES))
    random.shuffle(cromosoma)
    return cromosoma


def fitness(cromosoma: list) -> float:
    return 1.0 / longitud_ruta(cromosoma)


def seleccion_torneo(poblacion: list, fitnesses: list) -> list:
    """Elige TAM_TORNEO individuos al azar y devuelve el de mayor fitness."""
    competidores = random.sample(range(len(poblacion)), TAM_TORNEO)
    ganador = max(competidores, key=lambda k: fitnesses[k])
    return poblacion[ganador]


def crossover_ciclico(p1: list, p2: list) -> tuple:
    """
    Crossover cíclico (CX):
      1) Se arranca en la posición 0 y se toma el gen del padre 1.
      2) Se mira qué gen tiene el padre 2 en esa posición, se busca ese gen
         en el padre 1 y se salta a su posición. Se repite hasta volver al inicio
         (se cerró un ciclo).
      3) Las posiciones del ciclo se copian del padre 1; el resto, del padre 2.
    El segundo hijo se arma al revés. Siempre se obtienen permutaciones válidas
    y cada ciudad conserva la posición que tenía en alguno de los padres.
    """
    n = len(p1)
    pos_en_p1 = {gen: i for i, gen in enumerate(p1)}
    en_ciclo = [False] * n
    i = 0
    while not en_ciclo[i]:
        en_ciclo[i] = True
        i = pos_en_p1[p2[i]]
    h1 = [p1[k] if en_ciclo[k] else p2[k] for k in range(n)]
    h2 = [p2[k] if en_ciclo[k] else p1[k] for k in range(n)]
    return h1, h2


def mutacion(cromosoma: list) -> list:
    """
    Mutación por inversión: se eligen dos posiciones al azar y se invierte el
    tramo de genes entre ellas. Ej: [1 2 |3 4 5| 6] -> [1 2 |5 4 3| 6]
    En el mapa equivale a "descruzar" dos tramos del recorrido, por eso funciona
    mucho mejor que intercambiar dos genes sueltos.
    """
    c = cromosoma[:]
    if random.random() < PM:
        i, j = sorted(random.sample(range(N_CIUDADES), 2))
        c[i:j + 1] = c[i:j + 1][::-1]
    return c


def algoritmo_genetico(mostrar_progreso: bool = True) -> tuple:
    """
    Ciclo del AG:
      evaluar -> elitismo -> selección (torneo) -> crossover cíclico -> mutación
    Devuelve el mejor cromosoma encontrado y el historial (mín, prom, máx) por ciclo.
    """
    poblacion = [crear_cromosoma() for _ in range(N_POB)]
    mejor_global, mejor_d = None, float("inf")
    historial = []

    for ciclo in range(M_CICLOS):
        distancias = [longitud_ruta(c) for c in poblacion]
        fitnesses = [1.0 / d for d in distancias]

        # Estadísticas del ciclo
        historial.append((min(distancias), sum(distancias) / N_POB, max(distancias)))
        k_mejor = distancias.index(min(distancias))
        if distancias[k_mejor] < mejor_d:
            mejor_global, mejor_d = poblacion[k_mejor][:], distancias[k_mejor]
        if mostrar_progreso and (ciclo % 20 == 0 or ciclo == M_CICLOS - 1):
            print(f"    Ciclo {ciclo + 1:3d}: mejor = {historial[-1][0]:9,.1f} km | "
                  f"promedio = {historial[-1][1]:9,.1f} km")

        # Elitismo: los mejores pasan sin cambios
        orden = sorted(range(N_POB), key=lambda k: distancias[k])
        nueva = [poblacion[k][:] for k in orden[:ELITISMO]]

        # Resto de la población: selección + crossover + mutación
        while len(nueva) < N_POB:
            p1 = seleccion_torneo(poblacion, fitnesses)
            p2 = seleccion_torneo(poblacion, fitnesses)
            if random.random() < PC:
                h1, h2 = crossover_ciclico(p1, p2)
            else:
                h1, h2 = p1[:], p2[:]
            nueva.append(mutacion(h1))
            if len(nueva) < N_POB:
                nueva.append(mutacion(h2))
        poblacion = nueva

    return mejor_global, historial


def graficar_evolucion(historial: list, archivo: str = None):
    ciclos = range(1, len(historial) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(ciclos, [h[0] for h in historial], label="Mínimo (mejor)", color="#2E4A7D")
    plt.plot(ciclos, [h[1] for h in historial], label="Promedio", color="#f28e2b")
    plt.plot(ciclos, [h[2] for h in historial], label="Máximo (peor)", color="#bab0ac")
    plt.xlabel("Ciclo")
    plt.ylabel("Longitud del recorrido (km)")
    plt.title(f"Evolución del AG (N={N_POB}, M={M_CICLOS}, Pc={PC}, Pm={PM})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if archivo:
        plt.savefig(os.path.join(CARPETA, archivo), dpi=150)
        print(f"  Gráfico guardado en: {archivo}")
    plt.show()


def opcion_c():
    print(f"\n  Algoritmo genético: N={N_POB}, M={M_CICLOS}, Pc={PC}, Pm={PM}, "
          f"torneo={TAM_TORNEO}, elitismo={ELITISMO}\n")
    t0 = time.perf_counter()
    mejor, historial = algoritmo_genetico()
    print(f"\n  Tiempo de ejecución: {time.perf_counter() - t0:.2f} s")
    # Se muestra partiendo de La Plata (el recorrido es cerrado, el largo es el mismo)
    mejor = rotar_desde(mejor, 0)
    mostrar_ruta(mejor, "Recorrido mínimo encontrado con el algoritmo genético")
    print("  Cromosoma (genes 1-23):", [g + 1 for g in mejor])
    graficar_evolucion(historial, "evolucion_ag.png")
    dibujar_mapa(mejor, "Algoritmo genético", "mapa_ag.png")


# ─────────────────────────────────────────────
#  MENÚ PRINCIPAL
# ─────────────────────────────────────────────

def menu():
    while True:
        print("\n" + "=" * 62)
        print("  PROBLEMA DEL VIAJANTE - Capitales de Argentina")
        print("=" * 62)
        print("  1) Análisis del método exhaustivo")
        print("  a) Heurística desde una capital elegida")
        print("  b) Mejor recorrido con la heurística")
        print("  c) Algoritmo genético")
        print("  0) Salir")
        op = input("  Opción: ").strip().lower()
        if op == "1":
            analisis_exhaustivo()
        elif op == "a":
            opcion_a()
        elif op == "b":
            opcion_b()
        elif op == "c":
            opcion_c()
        elif op == "0":
            break
        else:
            print("  Opción inválida.")


if __name__ == "__main__":
    menu()

#from Rhino.Geometry import *
from typing import List

import math

class Vector3d:
    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z

    def VectorAngle(self, vector1, vector2):
        a = vector1.X * vector2.X + vector1.Y * vector2.Y + vector1.Z * vector2.Z
        b = vector1.Length * vector2.Length
        return math.acos(a/b)

    @property
    def Length(self):
        return math.sqrt(self.X ** 2 + self.Y ** 2 + self.Z ** 2)
###########################




class OpaqueMaterial:
    def __init__(self, name="Default Concrete", conductivity=2.0, density=2400, reflectivity=0.5, absorbance=0.3,
                 heat_capacity=1.0):
        self.name = name
        self.conductivity = conductivity
        self.density = density
        self.reflectivity = reflectivity
        self.absorbance = absorbance
        self.heat_capacity = heat_capacity

    def GetThermalResistance(self, thickness):
        return thickness / self.conductivity

    def __str__(self):
        return self.name


#====================================#



class OpaqueConstructionMaterial:
    def __init__(self, material: OpaqueMaterial, thickness = 0.1, area = 1, temperature = 20):
        self.material = material
        self.thickness = thickness
        self.area = area
        self.temperature = temperature

        self.resistance = self.material.GetThermalResistance(self.thickness)
        self.q = self.temperature * self.area * self.thickness * self.material.density * self.material.heat_capacity
#====================================#


class Construction:
    def __init__(self, name = "Default Opaque Construction", area = 1.0, normal = Vector3d(0, -1, 0), height = 1.0):
        self.name = name
        self.materials = []
        self.normal = normal
        self.area = area
        self.height = height

        #GLOBAL OVERRIDES INITIALIZATION FOR INTERNAL TESTS ONLY
        self.U_value = 4.0
        self.absorbance = 0.3
        self.reflectivity = 0.7
        self.transmittance = 0.0


    def AddLayer(self, material: OpaqueConstructionMaterial):
        self.materials.append(material)


    def SetTestValues(self, U, A, R, T):
        self.U_value = U
        self.absorbance = A
        self.reflectivity = R
        self.transmittance = T

    def GetHeatlossCoeff(self):
        return self.area * self.U_value

    def GetUValue(self):
        return self.U_value

    def GetReflectivity(self, incident_angle):
        # TEST
        # TODO
        return self.reflectivity

    def GetTransmittance(self, incident_angle):
        # TEST
        # TODO
        return self.transmittance

    def GetAbsorbance(self, incident_angle):
        # TEST
        # TODO
        return self.absorbance

    def GetAverageReflectivity(self):
        # TODO
        return self.reflectivity

    def GetAverageAbsorbance(self):
        # TODO
        return self.absorbance

    def GetAverageTransmittance(self):
        # TODO
        return self.transmittance


#====================================#











class ProceduralRadiance:

    def __init__(self, G_dir, G_diff, refl, altitude, azimuth, normal: Vector3d):
        self.G_dir = G_dir
        self.G_diff = G_diff
        self.refl = refl
        self.altitude = math.radians(altitude)
        self.azimuth = math.radians(azimuth)
        self.normal = normal
        self.sun_vector = self.GetSunVector()

    def Run(self):
        self.alpha = self.GetAlpha()
        self.dir_factor = self.GetDirFactor()
        self.g_dir = self.GetDir()
        self.beta = self.GetBeta()
        self.gamma = self.GetGamma()
        self.diff_factor = self.GetDiffFactor()
        self.refl_factor = self.GetReflFactor()
        self.g_diff = self.GetDiff()
        self.g_refl = self.GetRefl()
        self.g = self.GetG()

    def GetSunVector(self):
        if self.altitude == 0:
            return Vector3d(0, -1, 0)
        else:
            x = math.cos(self.altitude) * math.sin(self.azimuth)
            y = math.cos(self.altitude) * math.cos(self.azimuth)
            z = math.sin(self.altitude)
            return Vector3d(x, y, z)

    def GetAlpha(self):
        return abs(self.sun_vector.VectorAngle(self.normal, self.sun_vector))

    def GetDirFactor(self):
        if self.altitude == 0:
            return 0
        else:
            if math.cos(self.alpha) <= 0:
                return 0
            else:
                return math.cos(self.alpha)

    def GetDir(self):
        return self.G_dir * self.dir_factor

    def GetBeta(self):
        return math.sin(self.altitude)

    def GetGamma(self):
        return abs(self.normal.VectorAngle(self.normal, Vector3d(0, 0, 1)))

    def GetDiffFactor(self):
        return (1 + math.cos(self.gamma)) / 2

    def GetReflFactor(self):
        return (1 - math.cos(self.gamma)) / 2

    def GetDiff(self):
        return self.G_diff * self.diff_factor

    def GetRefl(self):
        return self.refl_factor * self.refl * (self.G_diff + self.G_dir * math.sin(self.beta))

    def GetG(self):
        return self.g_refl + self.g_diff + self.g_dir

    def GetAnisotropicRadiation(self):
        return self.g_dir

    def GetIsotropicRadiation(self):
        return self.g_diff + self.g_refl

class Fresnel:
    def __init__(
            self, refraction_index_1=1.0003,
            refraction_index_2=1.52,
            incident_angle=1 / 6 * math.pi,
            s_polarized_ratio=0.5,
            incident_energy=1,
            backward_emission_energy=0.2,
            material_1_name="Generic Air",
            material_2_name="Generic Glass",
    ):
        self.refraction_index_1 = refraction_index_1
        self.refraction_index_2 = refraction_index_2
        self.incident_angle = incident_angle
        self.s_polarized_ratio = s_polarized_ratio
        self.p_polarized_ratio = 1 - s_polarized_ratio
        self.I = incident_energy
        self.E = backward_emission_energy
        self.material_1_name = material_1_name
        self.material_2_name = material_2_name

        self.graph_lines = []
        self.length = 2

        self.refl()
        self.energy()
        self.graph()

    def refl(self):
        ct = math.cos(self.incident_angle)
        n1 = self.refraction_index_1
        n2 = self.refraction_index_2
        p2 = math.sqrt(abs(1 - (n1 / n2 * math.sin(self.incident_angle)) ** 2))
        rs = ((n1 * ct - n2 * p2) / (n1 * ct + n2 * p2)) ** 2

        rp = ((n1 * p2 - n2 * ct) / (n1 * p2 + n2 * ct)) ** 2
        self.reflectivity = self.s_polarized_ratio * rs + self.p_polarized_ratio * rp
        self.refl_s = rs
        self.refl_p = rp
        self.transmit_angle = abs(math.asin(n1 / n2 * math.sin(self.incident_angle)))
        ctb = math.cos(self.transmit_angle)

        p2b = math.sqrt(abs(1 - (n2 / n1 * math.sin(self.transmit_angle)) ** 2))
        rsb = ((n2 * ctb - n1 * p2b) / (n2 * ctb + n1 * p2b)) ** 2
        rpb = ((n2 * p2b - n1 * ctb) / (n2 * p2b + n1 * ctb)) ** 2
        self.backward_reflectivity = self.s_polarized_ratio * rsb + self.p_polarized_ratio * rpb
        self.back_refl_s = rsb
        self.back_refl_p = rpb
        return

    def energy(self):
        self.TI = self.I * (1 - self.reflectivity)
        self.TE = self.E * self.backward_reflectivity
        self.RI = self.I * self.reflectivity
        self.RE = self.E * (1 - self.backward_reflectivity)
        self.T = self.TI + self.TE
        self.R = self.RI + self.RE
        return

class Weather:
    # An Hourly Record from the Observed Climatic Records
    def __init__(self, index, temperature, azimuth, altitude, direct_radiation, diffuse_radiation, wind_speed, wind_direction, pressure):
        self.index = index
        self.temperature = temperature
        self.azimuth = azimuth
        self.altitude = altitude
        self.direct_radiation = direct_radiation
        self.diffuse_radiation = diffuse_radiation
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.pressure = pressure

    def GetProceduralRadianceObject(self):
        return ProceduralRadiance(self.direct_radiation, self.diffuse_radiation, 0.3, self.altitude, self.azimuth, Vector3d(0, -1, 0))

class WeatherSet:
    def __init__(self):
        self.records = []

    def AddRecord(self, weather: Weather):
        self.records.append(weather)

    def GetRecord(self, index) -> Weather:
        return self.records[index]

class EnergyReq:
    def __init__(self, heating_load=0, cooling_load=0, passive_heating=0, passive_cooling=0, passive_ventilation=0):
        self.heating_load = heating_load
        self.cooling_load = cooling_load
        self.passive_heating = passive_heating
        self.passive_cooling = passive_cooling
        self.passive_ventilation = passive_ventilation

class Strategy:
    def __init__(self, opening_ratio=1.0, shading_ratio=0.05, expose_interior=False, conditioned_temperature=20.0):
        self.opening_ratio = opening_ratio
        self.shading_ratio = shading_ratio
        self.expose_interior = expose_interior
        self.conditioned_temperature = conditioned_temperature

    def __str__(self):
        # TODO: describe different strategies in strings
        pass

class StrategySet:
    def __init__(self):
        self.strategies = []

    def AddStrategy(self, strategy: Strategy):
        self.strategies.append(strategy)

    def GetStrategy(self, index):
        return self.strategies[index]

    def SetStrategy(self, strategy: Strategy, index):
        self.strategies[index] = strategy

    def SetGlobalConditionedTemperature(self, temperature):
        for strategy in self.strategies:
            strategy.conditioned_temperature = temperature

    # TODO add other useful methods to create strategy sets differently

class Exterior:
    def __init__(self, weather: Weather, constructions: List[Construction]):
        self.weather = weather
        self.constructions = constructions

        self.area = 0
        for construction in self.constructions:
            self.area += construction.area

class Wintergarden:
    def __init__(self, weather: Weather, constructions: List[Construction], heat_capacity, maximum_opening, offset):
        self.weather = weather
        self.heat_capacity = heat_capacity
        self.maximum_opening = maximum_opening
        self.offset = offset

        self.opening_ratio = 0.8
        self.opening = self.opening_ratio * self.maximum_opening
        self.constructions = constructions
        self.pending_heat = 0.0

        self.area = 0
        for construction in self.constructions:
            self.area += construction.area

    def ApplyStrategy(self, strategy: Strategy):
        self.opening_ratio = strategy.opening_ratio
        self.opening = self.opening_ratio * self.maximum_opening

    def SetFloorConstruction(self, construction: Construction):
        self.floor_construction = construction


class Interior:
    def __init__(self, weather: Weather, constructions: List[Construction]):
        self.weather = weather

        self.temperature = 20.0
        self.shading_ratio = 0.05
        self.expose_interior = False

        self.constructions = constructions
        self.pending_heat = 0.0

        self.area = 0
        for construction in self.constructions:
            self.area += construction.area

    def ApplyStrategy(self, strategy: Strategy):
        self.shading_ratio = strategy.shading_ratio
        self.expose_interior = strategy.expose_interior
        if not self.expose_interior:
            self.temperature = strategy.conditioned_temperature


class Model:
    def __init__(self, exterior: Exterior, wintergarden: Wintergarden, interior: Interior, weatherset: WeatherSet, strategyset: StrategySet, starting_hour = 0):
        self.exterior = exterior
        self.wintergarden = wintergarden
        self.interior = interior
        self.weatherset = weatherset
        self.weather = weatherset.GetRecord(0)
        self.strategyset = strategyset
        self.strategy = strategyset.GetStrategy(0)
        self.hour = starting_hour

    def UpdateModel(self):
        self.UpdateStrategy()
        self.UpdateWeather()
        self.UpdateSolarGain()
        self.UpdateVentilationLoss()
        self.UpdateInfiltrationLoss()
        self.UpdateInternalGain()
        self.UpdateHeatTransfer()
        self.UpdateVentilationRate()
        self.UpdateEnergyRequirements()

    def UpdateStrategy(self):
        self.strategy = self.strategyset.GetStrategy(self.hour)
        self.wintergarden.ApplyStrategy(self.strategy)
        self.interior.ApplyStrategy(self.strategy)

    def UpdateWeather(self):
        self.weather = self.weatherset.GetRecord(self.hour)
        self.exterior.weather = self.weather
        self.wintergarden.weather = self.weather
        self.interior.weather = self.weather
        self.procedural_radiance_object = self.weather.GetProceduralRadianceObject()

    def UpdateSolarGain(self):

        wintergarden_absorbed_radiation = 0.0
        inner_screen_anisotropic_radiation = 0.0
        inner_screen_isotropic_radiation = 0.0
        interior_absorbed_radiation = 0.0

        for construction in self.exterior.constructions:
            self.procedural_radiance_object.normal = construction.normal
            self.procedural_radiance_object.Run()
            ani_rad = self.procedural_radiance_object.GetAnisotropicRadiation() * construction.area
            iso_rad = self.procedural_radiance_object.GetIsotropicRadiation() * construction.area

            if ani_rad > 0:
                # ansotropic radiation transmitted into the wintergarden
                ani_rad = construction.GetTransmittance(self.procedural_radiance_object.alpha) * ani_rad
                # anisotropic radiation absorbed in the fenestrations on the screen
                wintergarden_absorbed_radiation += construction.GetAbsorbance(self.procedural_radiance_object.alpha) * ani_rad
            else:
                ani_rad = 0
            # isotropic radiation transmitted into the wintergarden
            iso_rad = construction.GetAverageTransmittance() * iso_rad
            # isotropic radiation absorbed in the fenestrations on the screen
            wintergarden_absorbed_radiation += construction.GetAverageAbsorbance() * iso_rad

            # angle of the global ZAxis and the fenestration normal
            angle = construction.normal.VectorAngle(construction.normal, Vector3d(0, 0, 1))
            if angle >= math.pi / 2:
                angle = math.pi - angle

            # ratio of the radiation from fenestrations hitting firstly the wintergarden floor
            wintergarden_floor_viewfactor = self.wintergarden.offset * math.cos(angle) / construction.height
            if wintergarden_floor_viewfactor > 1:
                wintergarden_floor_viewfactor = 1

            # anisotropic radiation absorbed by the wintergarden floor
            wintergarden_absorbed_radiation += self.wintergarden.floor_construction.GetAbsorbance(math.pi / 2 - self.procedural_radiance_object.altitude) * ani_rad * wintergarden_floor_viewfactor

            # anisotropic radiation incident onto inner envelope
            ani_rad = (1 - wintergarden_floor_viewfactor) * ani_rad

            # isotropic radiation absorbed by the wintergarden floor
            wintergarden_absorbed_radiation += self.wintergarden.floor_construction.GetAverageAbsorbance() * iso_rad * wintergarden_floor_viewfactor

            # isotropic radiation reflected by the wintergarden floor, half of which contributes to isotropic radiation incident onto inner envelope
            iso_rad_bypass = iso_rad * (1 - wintergarden_floor_viewfactor)
            iso_rad = self.wintergarden.floor_construction.GetAverageReflectivity() * iso_rad / 2 * wintergarden_floor_viewfactor

            # anisotropic radiation reflected by the wintergarden floor, half of which contributes to isotropic radiation incident onto inner envelope
            iso_rad += self.wintergarden.floor_construction.GetReflectivity(math.pi / 2 - self.procedural_radiance_object.altitude) * ani_rad / 2 * wintergarden_floor_viewfactor

            # isotropic radiation directly incident on inner envelope
            iso_rad += iso_rad_bypass

            inner_screen_isotropic_radiation += iso_rad
            inner_screen_anisotropic_radiation += ani_rad

        for construction in self.wintergarden.constructions:
            self.procedural_radiance_object.normal = construction.normal
            self.procedural_radiance_object.Run()

            local_anisotropic_radiation = inner_screen_anisotropic_radiation * construction.area / self.wintergarden.area
            local_isotropic_radiation = inner_screen_isotropic_radiation * construction.area / self.wintergarden.area

            # interior radiation gain from transmitted anisotropic radiation
            interior_absorbed_radiation += local_anisotropic_radiation * construction.GetTransmittance(self.procedural_radiance_object.alpha)

            # wintergarden radiation gain from inner screen reflection
            wintergarden_absorbed_radiation += self.wintergarden.offset / math.pi / 2 * (local_isotropic_radiation * construction.GetAverageReflectivity())
            wintergarden_absorbed_radiation += self.wintergarden.offset / math.pi / 2 * (local_anisotropic_radiation * construction.GetReflectivity(self.procedural_radiance_object.alpha))

            # interior radiation gain from transmitted isotropic radiation
            interior_absorbed_radiation += local_isotropic_radiation * construction.GetAverageTransmittance()


        print(wintergarden_absorbed_radiation)
        print(interior_absorbed_radiation)


    def UpdateVentilationLoss(self):
        #TODO
        pass

    def UpdateInfiltrationLoss(self):
        #TODO
        pass

    def UpdateInternalGain(self):
        # TODO
        pass

    def UpdateHeatTransfer(self):
        # TODO
        pass

    def UpdateVentilationRate(self):
        #TODO
        pass

    def UpdateEnergyRequirements(self):
        #TODO
        pass




















# ====================================#

north_wall = Construction(name = "North Facade Exterior Wall", area = 5)
north_aperture = Construction(name = "North Facade Window", area = 3)
screen1 = Construction(name = "Wintergarden Glazing", area = 32, height = 8)
screen1.SetTestValues(1, 0.1, 0.1, 0.8)
screen2 = Construction(name = "Wintergarden Glazing", area = 32, height = 8)
screen2.SetTestValues(1, 0.3, 0.3, 0.4) #UART
inner_screen = Construction(name = "Interior Screen", area = 48, height = 6)
inner_screen.SetTestValues(1, 0, 0.9, 0.1)
wintergarden_floor = Construction(name = "Wintergarden Floor", area = 1.2 * 8)
wintergarden_floor.SetTestValues(1, 0.3, 0.7, 0)


test_weather = Weather(0, 20, 120, 30, 600, 120, 3, 60, 101204)
wintergarden = Wintergarden(test_weather, [inner_screen], 1, 2, 1.2)
wintergarden.SetFloorConstruction(wintergarden_floor)
exterior = Exterior(test_weather, [screen1, screen2])
interior = Interior(test_weather, [north_wall])

weatherset = WeatherSet()
weatherset.AddRecord(test_weather)

strategy = Strategy()
strategyset = StrategySet()
strategyset.AddStrategy(strategy)


test_model = Model(exterior, wintergarden, interior, weatherset, strategyset)
test_model.UpdateWeather()
test_model.UpdateSolarGain()
'''
p1 = ProceduralRadiance(600, 100, 0.3, 30, 120, Vector3d(0, -1, 0))
p1.Run()
print (p1.g_dir)
p2 = ProceduralRadiance(200, 100, 0.3, 30, 120, Vector3d(0, -1, 0))
p2.Run()
print (p2.g_dir)
'''
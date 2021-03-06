#from Rhino.Geometry import *
from typing import List
import math
import time
import os
import matplotlib.pyplot as plt
import numpy


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
    def __init__(self, name = "Default Opaque Construction", area = 1.0, normal = Vector3d(0, -1, 0), height = 1.0, thickness = 0.0,  density = 1.5, U = 4.0):
        self.name = name
        self.materials = []
        self.normal = normal
        self.area = area
        self.height = height
        self.thickness = thickness
        self.density = density
        self.U = U
        self.U_multiplier = 1.0
        self.hour = 0

        #GLOBAL OVERRIDES INITIALIZATION FOR INTERNAL TESTS ONLY
        self.heat_capacity = 600 / 3600

        self.absorbance = 0.3
        self.reflectivity = 0.7
        self.transmittance = 0.0


    def AddLayer(self, material: OpaqueConstructionMaterial):
        self.materials.append(material)
        self.thickness += material.thickness
        # TODO
        # MASS

    def SetTestValues(self, U, A, R, T):
        self.U = U
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

    @property
    def Mass(self):
        return self.thickness * self.area * self.density

    @property
    def HeatExchangeCoeff(self):
        return self.U_value * self.area

    @property
    def HeatCapacity(self):
        return self.heat_capacity

    @property
    def U_value(self):
        return self.U * self.U_multiplier

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

    def GetExteriorAirHeatCapacity(self):
        return 1005.0 / 3600

    def GetAirDensity(self):
        # EXPERIMENTAL ESTIMATIONS
        return 1.225



class WeatherSet:
    def __init__(self):
        self.records = []

    def AddRecord(self, weather: Weather):
        self.records.append(weather)

    def GetRecord(self, index) -> Weather:
        return self.records[index]

    @property
    def Length(self):
        return len(self.records)

    def IsSummerTime(self, hour):
        if hour <= 6500 and hour >= 3600:
            return True
        else:
            return False

    def IsDayTime(self, hour):
        hod = hour - 24 * (math.floor(hour / 24))
        if hod >= 6 and hod <= 18:
            return True
        else:
            return False


class Strategy:
    def __init__(self, hour: int=0, opening_ratio=1.0, interior_opening_ratio=0.02, shading_ratio=0.05, north_opening_ratio = 0.05, heating_threshold = 10.0, heating_goal = 15.0, cooling_goal = 25.0, cooling_threshold = 32.0, internal_gain = 1.2):
        self.hour = hour
        self.opening_ratio = opening_ratio
        self.interior_opening_ratio = interior_opening_ratio
        self.north_opening_ratio = north_opening_ratio
        self.shading_ratio = shading_ratio
        self.heating_threshold = heating_threshold
        self.heating_goal = heating_goal
        self.cooling_threshold = cooling_threshold
        self.cooling_goal = cooling_goal
        self.internal_gain = internal_gain
        self.nightpurge_start = 3600
        self.nightpurge_end = 7000

        self.summer_window_u_value_multiplier = 1.0
        self.summer_night_purge = False
        self.summer_shading = False
        self.summer_ventilation = False
        self.enclose_wintergarden = False

    @property
    def IsDayTime(self):
        ws = WeatherSet()
        return ws.IsDayTime(self.hour)

    @property
    def IsSummerTime(self):
        ws = WeatherSet()
        return ws.IsSummerTime(self.hour)

    def __str__(self):
        # TODO: describe different strategies in strings
        pass

    def DoSummerNightPurge(self, purge: bool):
        self.summer_night_purge = purge

    def SetSummerWindowUValueMultiplier(self, param):
        self.summer_window_u_value_multiplier = param

    def DoSummerShading(self, shade: bool):
        self.summer_shading = shade

    def DoSummerVentilation(self, vent: bool):
        self.summer_ventilation = vent

    def DoEncloseWintergarden(self, enclose: bool):
        self.enclose_wintergarden = enclose

    @property
    def UMultipier(self):
        if self.IsSummerTime:
            return self.summer_window_u_value_multiplier
        else:
            return 1.0

    @property
    def MaximumVentilation(self):
        if self.summer_ventilation and self.IsSummerTime:
            return True
        """
        if self.summer_night_purge and self.IsSummerTime:
            if self.IsDayTime:
                return False
            else:
                return True
        """
        return False

    @property
    def Shading(self):
        if self.IsSummerTime and self.summer_shading:
            return 1.0
        else:
            return self.shading_ratio


class StrategySet:
    def __init__(self):
        self.strategies = []

    def AddStrategy(self, strategy: Strategy):
        self.strategies.append(strategy)

    def GetStrategy(self, index) -> Strategy:
        return self.strategies[index]

    def SetStrategy(self, strategy: Strategy, index):
        self.strategies[index] = strategy

    # TODO add other useful methods to create strategy sets differently


class Exterior:
    def __init__(self, weather: Weather, constructions: List[Construction]):
        self.weather = weather
        self.constructions = constructions

        self.area = 0
        for construction in self.constructions:
            self.area += construction.area

    @property
    def HeatExchangeCoeff(self):
        coeff = 0.0
        for construction in self.constructions:
            coeff += construction.HeatExchangeCoeff
        return coeff


class Wintergarden:
    def __init__(self, weather: Weather, constructions: List[Construction], maximum_opening, offset):
        self.weather = weather
        self.maximum_opening = maximum_opening
        self.offset = offset

        self.opening_ratio = 0.2
        self.shading_ratio = 0.0
        self.opening = self.opening_ratio * self.maximum_opening
        self.constructions = constructions

        self.height = 8.0

        self.pending_heat = 0.0

        self.temperature = 12
        self.exterior_ventilation_rate = 0.0
        self.exterior_infiltration_rate = 0.0
        self.interior_ventilation_rate = 0.0
        self.interior_infiltration_rate = 0.0

    def ApplyStrategy(self, strategy: Strategy):
        self.opening_ratio = strategy.opening_ratio
        if strategy.MaximumVentilation:
            self.opening_ratio = 1.0
        else:
            self.opening_ratio = 0.05
        self.opening = self.opening_ratio * self.maximum_opening
        self.shading_ratio = strategy.Shading
        for construction in self.constructions:
            if construction.GetAverageTransmittance() >= 0.1:
                construction.U_multiplier = strategy.UMultipier

        # TEST
        if strategy.enclose_wintergarden:
            pass
        else:
            self.temperature = self.weather.temperature



    def SetFloorConstruction(self, construction: Construction):
        self.floor_construction = construction

    def SetShadingConstruction(self, construction: Construction):
        self.shading_construction = construction

    @property
    def FaceArea(self):
        a = 0.0
        for construction in self.constructions:
            a += construction.area
        return a

    @property
    def Mass(self):
        mass = 0
        for construction in self.constructions:
            mass += construction.Mass
        mass += self.floor_construction.Mass
        return mass

    @property
    def HeatExchangeCoeff(self):
        coeff = 0.0
        for construction in self.constructions:
            coeff += construction.HeatExchangeCoeff
        return coeff

    @property
    def TotalHeatCapacity(self):
        hc = 0.0
        for construction in self.constructions:
            hc += construction.Mass * construction.HeatCapacity
        hc += self.shading_construction.Mass * self.shading_construction.heat_capacity
        hc += self.FaceArea * self.offset * self.weather.GetAirDensity() * self.weather.GetExteriorAirHeatCapacity()
        return hc

    @property
    def StackEffectPressureDifference(self):
        return abs(0.0342 * self.weather.pressure * self.height * (1 / (self.temperature + 273) - 1 / (self.weather.temperature + 273)))

    def StackEffectFlowRate(self, temperature):
        return 0.65 * self.opening * math.sqrt(abs(2 * 9.807 * self.height / 2 * (self.temperature - temperature) / (self.temperature + 273))) * 3600

    def GetVentilationHeatExchange(self, temperature, flow):

        total_heat = self.TotalHeatCapacity * (self.temperature - temperature)
        volume_heat = self.TotalHeatCapacity / (self.TotalHeatCapacity + self.weather.GetAirDensity() * self.weather.GetExteriorAirHeatCapacity() * flow) * total_heat
        return volume_heat - total_heat


class Interior:
    def __init__(self, weather: Weather, constructions: List[Construction], maximum_opening, offset):
        self.weather = weather

        self.shading_ratio = 0.05
        self.opening_ratio = 0.0

        self.constructions = constructions


        self.height = 6.5
        self.offset = offset
        self.maximum_opening = maximum_opening
        self.opening = self.maximum_opening * self.opening_ratio


        self.pending_heat = 0.0

        self.temperature = 12.0
        self.exterior_ventilation_rate = 0.0
        self.exterior_infiltration_rate = 0.0
        self.room_ventilation_rate = 2000
        self.room_infiltration_rate = 50


        self.heating_threshold = 10.0
        self.heating_goal = 20
        self.cooling_threshold = 35.0
        self.cooling_goal = 26.0
        self.internal_gain = 0.0

    def ApplyStrategy(self, strategy: Strategy):
        self.opening_ratio = strategy.interior_opening_ratio
        if strategy.MaximumVentilation:
            self.opening_ratio = 1.0
        else:
            self.opening_ratio = 0.05
        self.opening = self.opening_ratio * self.maximum_opening

        self.internal_gain = strategy.internal_gain * self.floor_construction.area
        self.heating_threshold = strategy.heating_threshold
        self.heating_goal = strategy.heating_goal
        self.cooling_threshold = strategy.cooling_threshold
        self.cooling_goal = strategy.cooling_goal

        for construction in self.constructions:
            if construction.GetAverageTransmittance() >= 0.1:
                construction.U_multiplier = strategy.UMultipier



    def SetFloorConstruction(self, construction: Construction):
        self.floor_construction = construction

    def SetPartyWallConstruction(self, construction: Construction):
        self.party_wall_construction = construction

    @property
    def FaceArea(self):
        a = 0.0
        for construction in self.constructions:
            a += construction.area
        return a

    @property
    def Mass(self):
        mass = 0.0
        for construction in self.constructions:
            mass += construction.Mass
        mass += self.floor_construction.Mass
        mass += self.party_wall_construction.Mass
        return mass

    @property
    def HeatExchangeCoeff(self):
        coeff = 0.0
        for construction in self.constructions:
            coeff += construction.HeatExchangeCoeff

        return coeff

    @property
    def TotalHeatCapacity(self):
        hc = 0.0
        for construction in self.constructions:
            hc += construction.Mass * construction.HeatCapacity
        hc += self.floor_construction.Mass * self.floor_construction.heat_capacity
        hc += self.party_wall_construction.Mass * self.party_wall_construction.heat_capacity
        hc += self.FaceArea * self.height * self.weather.GetAirDensity() * self.weather.GetExteriorAirHeatCapacity()
        return hc

    @property
    def StackEffectPressureDifference(self):
        return abs(0.0342 * self.weather.pressure * self.height * (1 / (self.temperature + 273) - 1 / (self.weather.temperature + 273)))

    def StackEffectFlowRate(self, temperature):

        return 0.65 * self.opening * math.sqrt(abs(2 * 9.807 * self.height / 2 * (self.temperature - temperature) / (self.temperature + 273))) * 3600

    def GetVentilationHeatExchange(self, temperature, flow):

        total_heat = self.TotalHeatCapacity * (self.temperature - temperature)
        volume_heat = self.TotalHeatCapacity / (self.TotalHeatCapacity + self.weather.GetAirDensity() * self.weather.GetExteriorAirHeatCapacity() * flow) * total_heat
        return volume_heat - total_heat


class Room:
    def __init__(self, weather: Weather, constructions: List[Construction], opening_ratio=0.0):
        self.weather = weather
        self.constructions = constructions

        self.opening_ratio = opening_ratio
        self.maximum_opening = 6.2
        self.opening = opening_ratio * self.maximum_opening

        self.height = 3.8
        self.offset = 4.4

        self.pending_heat = 0.0

        self.temperature = 15.0
        self.exterior_ventilation_rate = 0.0
        self.exterior_infiltration_rate = 0.0
        self.interior_ventilation_rate = 2000
        self.interior_infiltration_rate = 50

        self.heating_threshold = 10.0
        self.heating_goal = 20
        self.cooling_threshold = 35.0
        self.cooling_goal = 26.0
        self.internal_gain = 0.0

    def ApplyStrategy(self, strategy: Strategy):
        self.internal_gain = strategy.internal_gain * self.floor_construction.area
        self.opening_ratio = strategy.north_opening_ratio
        if strategy.MaximumVentilation:
            self.opening_ratio = 1.0
        else:
            self.opening_ratio = 0
        self.opening = self.opening_ratio * self.maximum_opening
        self.heating_threshold = strategy.heating_threshold
        self.heating_goal = strategy.heating_goal
        self.cooling_threshold = strategy.cooling_threshold
        self.cooling_goal = strategy.cooling_goal
        self.opening_ratio = strategy.north_opening_ratio
        self.opening = self.opening_ratio * self.maximum_opening

        for construction in self.constructions:
            if construction.GetAverageTransmittance() >= 0.1:
                construction.U_multiplier = strategy.UMultipier





    def SetFloorConstruction(self, construction: Construction):
        self.floor_construction = construction

    def SetPartyWallConstruction(self, construction: Construction):
        self.party_wall_construction = construction

    @property
    def FaceArea(self):
        a = 0.0
        for construction in self.constructions:
            a += construction.area
        return a

    @property
    def Mass(self):
        mass = 0.0
        for construction in self.constructions:
            mass += construction.Mass
        mass += self.floor_construction.Mass
        mass += self.party_wall_construction.Mass
        return mass

    @property
    def HeatExchangeCoeff(self):
        coeff = 0.0
        for construction in self.constructions:
            coeff += construction.HeatExchangeCoeff
        return coeff

    @property
    def TotalHeatCapacity(self):
        hc = 0.0
        for construction in self.constructions:
            hc += construction.Mass * construction.HeatCapacity
        hc += self.floor_construction.Mass * self.floor_construction.heat_capacity
        hc += self.party_wall_construction.Mass * self.party_wall_construction.heat_capacity
        hc += self.FaceArea * self.height * self.weather.GetAirDensity() * self.weather.GetExteriorAirHeatCapacity()
        return hc

    @property
    def StackEffectPressureDifference(self):
        return abs(0.0342 * self.weather.pressure * self.height * (1 / (self.temperature + 273) - 1 / (self.weather.temperature + 273)))

    def StackEffectFlowRate(self, temperature):
        return 0.65 * self.opening * math.sqrt(abs(2 * 9.807 * self.height / 2 * (self.temperature - temperature) / (self.temperature + 273))) * 3600

    def GetVentilationHeatExchange(self, temperature, flow):

        total_heat = self.TotalHeatCapacity * (self.temperature - temperature)
        volume_heat = self.TotalHeatCapacity / (self.TotalHeatCapacity + self.weather.GetAirDensity() * self.weather.GetExteriorAirHeatCapacity() * flow) * total_heat
        return volume_heat - total_heat


class Model:
    def __init__(self, exterior: Exterior, wintergarden: Wintergarden, interior: Interior, room: Room, weatherset: WeatherSet, strategyset: StrategySet, starting_hour = 0, analysis_period = -1):
        self.exterior = exterior
        self.wintergarden = wintergarden
        self.interior = interior
        self.room = room
        self.weatherset = weatherset
        self.weather = weatherset.GetRecord(0)
        self.strategyset = strategyset
        self.strategy = strategyset.GetStrategy(0)
        self.hour = starting_hour
        self.analysis_period = analysis_period
        self.ready = True
        self.results = SimulationResults()

    def UpdateModel(self):

        self.UpdateStrategy()
        self.UpdateWeather()
        self.UpdateSolarGain()
        self.UpdateVentilationLoss()
        self.UpdateLatentHeat()
        self.UpdateInternalGain()
        self.UpdateHeatTransfer()
        self.UpdateVentilationRate()
        self.UpdateResult()

        if self.interior.temperature <= 200:
            print(str(self.hour) + " " +str(self.exterior.weather.temperature) + " " + str(self.wintergarden.temperature) + " " +str(self.interior.temperature) + " " +str(self.room.temperature))

        if self.hour >= self.weatherset.Length - 1:
            self.ready = False
            return False

        if self.analysis_period != 0:
            self.analysis_period += -1
            self.hour += 1
            self.ready = True
            return True
        else:
            self.ready = False
            return False

    def UpdateStrategy(self):

        self.strategy = self.strategyset.GetStrategy(0)#self.hour#)
        self.strategy.hour = self.hour
        self.wintergarden.ApplyStrategy(self.strategy)
        self.interior.ApplyStrategy(self.strategy)
        self.room.ApplyStrategy(self.strategy)


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
        room_absorbed_radiation = 0.0

        # simulation result energy breakdowns
        wintergarden_radiation_from_anisotropic = 0.0
        wintergarden_radiation_from_isotropic = 0.0

        interior_radiation_from_anisotropic = 0.0
        interior_radiation_from_isotropic_south = 0.0

        interior_radiation_from_isotropic_north = 0.0



        for construction in self.exterior.constructions: # first screen layer
            self.procedural_radiance_object.normal = construction.normal
            self.procedural_radiance_object.Run()
            ani_rad = self.procedural_radiance_object.GetAnisotropicRadiation() * construction.area
            iso_rad = self.procedural_radiance_object.GetIsotropicRadiation() * construction.area

            if ani_rad > 0:
                # anisotropic radiation transmitted into the wintergarden
                ani_rad = construction.GetTransmittance(self.procedural_radiance_object.alpha) * ani_rad
                # anisotropic radiation absorbed in the fenestrations on the screen
                wintergarden_absorbed_radiation += construction.GetAbsorbance(self.procedural_radiance_object.alpha) * ani_rad
                # update to simulation result energy breakdowns
                wintergarden_radiation_from_anisotropic += construction.GetAbsorbance(self.procedural_radiance_object.alpha) * ani_rad
            else:
                ani_rad = 0
            # isotropic radiation transmitted into the wintergarden
            iso_rad = construction.GetAverageTransmittance() * iso_rad
            # isotropic radiation absorbed in the fenestrations on the screen
            wintergarden_absorbed_radiation += construction.GetAverageAbsorbance() * iso_rad
            # update to simulation result energy breakdowns
            wintergarden_radiation_from_isotropic += construction.GetAverageAbsorbance() * iso_rad

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
            # update to simulation result energy breakdowns
            wintergarden_radiation_from_anisotropic += self.wintergarden.floor_construction.GetAbsorbance(math.pi / 2 - self.procedural_radiance_object.altitude) * ani_rad * wintergarden_floor_viewfactor

            # anisotropic radiation incident onto inner envelope
            ani_rad = (1 - wintergarden_floor_viewfactor) * ani_rad

            # isotropic radiation absorbed by the wintergarden floor
            wintergarden_absorbed_radiation += self.wintergarden.floor_construction.GetAverageAbsorbance() * iso_rad * wintergarden_floor_viewfactor
            # update to simulation result energy breakdowns
            wintergarden_radiation_from_isotropic += self.wintergarden.floor_construction.GetAverageAbsorbance() * iso_rad * wintergarden_floor_viewfactor

            # isotropic radiation reflected by the wintergarden floor, half of which contributes to isotropic radiation incident onto inner envelope
            iso_rad_bypass = iso_rad * (1 - wintergarden_floor_viewfactor)
            iso_rad = self.wintergarden.floor_construction.GetAverageReflectivity() * iso_rad / 2 * wintergarden_floor_viewfactor

            # anisotropic radiation reflected by the wintergarden floor, half of which contributes to isotropic radiation incident onto inner envelope
            iso_rad += self.wintergarden.floor_construction.GetReflectivity(math.pi / 2 - self.procedural_radiance_object.altitude) * ani_rad / 2 * wintergarden_floor_viewfactor

            # isotropic radiation directly incident on inner envelope
            iso_rad += iso_rad_bypass

            inner_screen_isotropic_radiation += iso_rad
            inner_screen_anisotropic_radiation += ani_rad

        # shading
        self.procedural_radiance_object.normal = self.wintergarden.shading_construction.normal
        self.procedural_radiance_object.Run()


        portion_shaded = self.wintergarden.shading_construction.area / self.wintergarden.FaceArea * self.wintergarden.shading_ratio
        shading_radiation_from_isotropic = 0.0
        shading_radiation_from_anisotropic = 0.0

        shading_radiation_from_anisotropic += portion_shaded * inner_screen_anisotropic_radiation * self.wintergarden.shading_construction.GetReflectivity(self.procedural_radiance_object.alpha) * self.wintergarden.offset / math.pi / 2
        shading_radiation_from_anisotropic += portion_shaded * inner_screen_anisotropic_radiation * self.wintergarden.shading_construction.GetAbsorbance(self.procedural_radiance_object.alpha) * self.wintergarden.offset / math.pi / 2
        shading_radiation_from_isotropic += portion_shaded * inner_screen_isotropic_radiation * self.wintergarden.shading_construction.GetAverageAbsorbance()
        shading_radiation_from_isotropic += portion_shaded * inner_screen_isotropic_radiation * self.wintergarden.shading_construction.GetAverageReflectivity() * self.wintergarden.offset / math.pi / 2

        wintergarden_radiation_from_anisotropic += shading_radiation_from_anisotropic
        wintergarden_radiation_from_isotropic += shading_radiation_from_isotropic

        wintergarden_absorbed_radiation += shading_radiation_from_isotropic + shading_radiation_from_anisotropic


        # 0304 ADDED
        inner_screen_anisotropic_radiation += -shading_radiation_from_anisotropic
        inner_screen_isotropic_radiation += -shading_radiation_from_isotropic


        for construction in self.wintergarden.constructions: # second screen layer
            self.procedural_radiance_object.normal = construction.normal
            self.procedural_radiance_object.Run()

            local_anisotropic_radiation = inner_screen_anisotropic_radiation * construction.area / self.wintergarden.FaceArea
            local_isotropic_radiation = inner_screen_isotropic_radiation * construction.area / self.wintergarden.FaceArea

            # interior radiation gain from transmitted anisotropic radiation
            interior_absorbed_radiation += local_anisotropic_radiation * construction.GetTransmittance(self.procedural_radiance_object.alpha)
            # update to simulation result energy breakdowns
            interior_radiation_from_anisotropic += local_anisotropic_radiation * construction.GetTransmittance(self.procedural_radiance_object.alpha)

            # interior radiation absorbed from fabric material absorbance, isotropic sector
            interior_absorbed_radiation += local_isotropic_radiation * construction.GetAverageAbsorbance()
            # update to simulation result energy breakdowns
            interior_radiation_from_isotropic_south += local_isotropic_radiation * construction.GetAverageAbsorbance()
            # interior radiation absorbed from fabric material absorbance, anisotropic sector
            interior_absorbed_radiation += local_anisotropic_radiation * construction.GetAbsorbance(self.procedural_radiance_object.alpha)
            # update to simulation result energy breakdowns
            interior_radiation_from_anisotropic += local_anisotropic_radiation * construction.GetAbsorbance(self.procedural_radiance_object.alpha)

            # wintergarden radiation gain from inner screen reflection, isotropic sector
            wintergarden_absorbed_radiation += self.wintergarden.offset / math.pi / 2 * (local_isotropic_radiation * construction.GetAverageReflectivity())
            # update to simulation result energy breakdowns
            wintergarden_radiation_from_isotropic += self.wintergarden.offset / math.pi / 2 * (local_isotropic_radiation * construction.GetAverageReflectivity())

            # wintergarden radiation gain from inner screen reflection, anisotropic sector
            wintergarden_absorbed_radiation += self.wintergarden.offset / math.pi / 2 * (local_anisotropic_radiation * construction.GetReflectivity(self.procedural_radiance_object.alpha))
            # update to simulation result energy breakdowns
            wintergarden_radiation_from_anisotropic += self.wintergarden.offset / math.pi / 2 * (local_anisotropic_radiation * construction.GetReflectivity(self.procedural_radiance_object.alpha))

            # interior radiation gain from transmitted isotropic radiation
            interior_absorbed_radiation += local_isotropic_radiation * construction.GetAverageTransmittance()
            # update to simulation result energy breakdowns
            interior_radiation_from_isotropic_south += local_isotropic_radiation * construction.GetAverageTransmittance()


        for construction in self.room.constructions: # north facing facade
            self.procedural_radiance_object.normal = construction.normal
            self.procedural_radiance_object.Run()

            iso_rad = self.procedural_radiance_object.GetIsotropicRadiation() * construction.area

            # isotropic radiation transmitted into the interior
            room_absorbed_radiation += iso_rad * construction.GetAverageTransmittance()
            # update to simulation result energy breakdowns
            interior_radiation_from_isotropic_north += iso_rad * construction.GetAverageTransmittance()

            # isotropic radiation absorbed by the fabric
            room_absorbed_radiation += iso_rad * construction.GetAverageAbsorbance()
            # update to simulation result energy breakdowns
            interior_radiation_from_isotropic_north += iso_rad * construction.GetAverageAbsorbance()


        self.wintergarden.pending_heat += wintergarden_absorbed_radiation
        self.interior.pending_heat += interior_absorbed_radiation
        self.room.pending_heat += room_absorbed_radiation

        self.results.wintergarden_radiation_from_anisotropic.append(wintergarden_radiation_from_anisotropic)
        self.results.wintergarden_radiation_from_isotropic.append(wintergarden_radiation_from_isotropic)
        self.results.interior_radiation_from_isotropic_north.append(interior_radiation_from_isotropic_north)
        self.results.interior_radiation_from_isotropic_south.append(interior_radiation_from_isotropic_south)
        self.results.interior_radiation_from_anisotropic.append(interior_radiation_from_anisotropic)
        self.results.shading_radiation_from_anisotropic.append(shading_radiation_from_anisotropic)
        self.results.shading_radiation_from_isotropic.append(shading_radiation_from_isotropic)


        # TEST
        # print(wintergarden_absorbed_radiation)
        # print(interior_absorbed_radiation)

    def UpdateVentilationLoss(self):
        if self.hour >= 3598:
            pass

        # TODO REWORK ON THIS?
        self.wintergarden.exterior_ventilation_rate = self.wintergarden.StackEffectFlowRate(self.weather.temperature)
        self.wintergarden.interior_ventilation_rate = self.interior.StackEffectFlowRate(self.wintergarden.temperature)

        self.wintergarden.exterior_infiltration_rate = (self.weather.wind_speed * 0.02 + 0.02) * 3600
        self.wintergarden.interior_infiltration_rate = (self.weather.wind_speed * 0.01 + 0.01) * 3600

        self.interior.wintergarden_infiltration_rate = 0.02 * 3600
        self.wintergarden.interior_infiltration_rate = 0.02 * 3600

        self.room.exterior_ventilation_rate = (self.weather.wind_speed * 0.02 + 0.02) * 3600
        self.room.exterior_infiltration_rate = (self.weather.wind_speed * 0.01 + 0.01) * 3600


        ventilation_heat_exchange_ew = self.wintergarden.GetVentilationHeatExchange(self.weather.temperature, self.wintergarden.exterior_ventilation_rate)/2

        ventilation_heat_exchange_wi = self.interior.GetVentilationHeatExchange(self.wintergarden.temperature, self.wintergarden.interior_ventilation_rate)/2


        ventilation_heat_exchange_ir = self.room.GetVentilationHeatExchange(self.interior.temperature, self.interior.room_ventilation_rate)
        ventilation_heat_exchange_re = - self.interior.GetVentilationHeatExchange(self.weather.temperature, self.room.exterior_ventilation_rate)

        infiltration_heat_exchange_ew = self.wintergarden.GetVentilationHeatExchange(self.weather.temperature, self.wintergarden.exterior_infiltration_rate)
        infiltration_heat_exchange_wi = self.interior.GetVentilationHeatExchange(self.wintergarden.temperature, self.wintergarden.interior_infiltration_rate)
        infiltration_heat_exchange_ir = self.room.GetVentilationHeatExchange(self.interior.temperature, self.interior.room_infiltration_rate)
        infiltration_heat_exchange_re = - self.interior.GetVentilationHeatExchange(self.weather.temperature, self.room.exterior_infiltration_rate)


        self.wintergarden.pending_heat += ventilation_heat_exchange_ew - ventilation_heat_exchange_wi + infiltration_heat_exchange_ew - infiltration_heat_exchange_wi
        self.interior.pending_heat += ventilation_heat_exchange_wi - ventilation_heat_exchange_ir + infiltration_heat_exchange_wi - infiltration_heat_exchange_ir
        self.room.pending_heat += ventilation_heat_exchange_ir - ventilation_heat_exchange_re + infiltration_heat_exchange_ir - infiltration_heat_exchange_re

        self.results.ventilation_heat_exchange_ew.append(ventilation_heat_exchange_ew)
        self.results.ventilation_heat_exchange_wi.append(ventilation_heat_exchange_wi)
        self.results.ventilation_heat_exchange_ir.append(ventilation_heat_exchange_ir)
        self.results.ventilation_heat_exchange_re.append(ventilation_heat_exchange_re)

        self.results.infiltration_heat_exchange_ew.append(infiltration_heat_exchange_ew)
        self.results.infiltration_heat_exchange_wi.append(infiltration_heat_exchange_wi)
        self.results.infiltration_heat_exchange_ir.append(infiltration_heat_exchange_ir)
        self.results.infiltration_heat_exchange_re.append(infiltration_heat_exchange_re)

        return

        # ??????????????? OBSOLETE DO NOT USE THIS CODE FOR CONVECTION HEAT EXCHANGE BETWEEN VENUES
        # heat exchange in forms of VENTILATION, and INFILTRATION between the exterior and the wintergarden, between the wintergarden and the interior
        ventilation_heat_exchange_ew = self.wintergarden.exterior_ventilation_rate * self.exterior.weather.GetAirDensity() * (self.weather.temperature - self.wintergarden.temperature)
        ventilation_heat_exchange_wi = self.wintergarden.interior_ventilation_rate * self.exterior.weather.GetAirDensity() * (self.wintergarden.temperature - self.interior.temperature)
        ventilation_heat_exchange_ie = self.interior.exterior_ventilation_rate * self.exterior.weather.GetAirDensity() * (self.interior.temperature - self.weather.temperature)

        infiltration_heat_exchange_ew = self.wintergarden.exterior_infiltration_rate * self.exterior.weather.GetAirDensity() * (self.weather.temperature - self.wintergarden.temperature)
        infiltration_heat_exchange_wi = self.wintergarden.interior_infiltration_rate * self.exterior.weather.GetAirDensity() * (self.wintergarden.temperature - self.interior.temperature)
        infiltration_heat_exchange_ie = self.interior.exterior_infiltration_rate * self.exterior.weather.GetAirDensity() * (self.interior.temperature - self.weather.temperature)

        self.wintergarden.pending_heat += ventilation_heat_exchange_ew - ventilation_heat_exchange_wi + infiltration_heat_exchange_ew - infiltration_heat_exchange_wi
        self.interior.pending_heat += ventilation_heat_exchange_wi - ventilation_heat_exchange_ie + infiltration_heat_exchange_wi - infiltration_heat_exchange_ie

        self.results.ventilation_heat_exchange_ew.append(ventilation_heat_exchange_ew)
        self.results.ventilation_heat_exchange_wi.append(ventilation_heat_exchange_wi)
        self.results.ventilation_heat_exchange_ie.append(ventilation_heat_exchange_ie)

        self.results.infiltration_heat_exchange_ew.append(infiltration_heat_exchange_ew)
        self.results.infiltration_heat_exchange_wi.append(infiltration_heat_exchange_wi)
        self.results.infiltration_heat_exchange_ie.append(infiltration_heat_exchange_ie)
        # TODO
        # CONSIDER LATENT HEAT GAIN / PSYCHROMETRIC PROCESSES

    def UpdateLatentHeat(self):
        # TODO Psychrometric Processes
        pass

    def UpdateInternalGain(self):
        self.interior.pending_heat += self.strategy.internal_gain
        self.room.pending_heat += self.room.internal_gain


    def UpdateHeatTransfer(self):

        # wintergarden conduction with exterior

        self.wintergarden.pending_heat += self.exterior.HeatExchangeCoeff * (self.exterior.weather.temperature - self.wintergarden.temperature)
        self.results.conduction_ew.append(self.exterior.HeatExchangeCoeff * (- self.exterior.weather.temperature + self.wintergarden.temperature))

        # wintergarden conduction with interior, bilateral rec
        self.wintergarden.pending_heat += self.wintergarden.HeatExchangeCoeff * (self.interior.temperature - self.wintergarden.temperature)
        self.interior.pending_heat += self.wintergarden.HeatExchangeCoeff * (self.wintergarden.temperature - self.interior.temperature)
        self.results.conduction_wi.append(self.wintergarden.HeatExchangeCoeff * (self.interior.temperature - self.wintergarden.temperature))

        # convert wintergarden pending heat into temperature differences
        self.wintergarden.temperature += self.wintergarden.pending_heat / self.wintergarden.TotalHeatCapacity

        self.wintergarden.pending_heat = 0


        # interior conduction with room, bilateral rec
        self.interior.pending_heat += self.interior.party_wall_construction.U_value * self.interior.party_wall_construction.area * (self.room.temperature - self.interior.temperature)
        self.room.pending_heat += self.interior.party_wall_construction.U_value * self.interior.party_wall_construction.area * (self.interior.temperature - self.room.temperature)
        self.results.conduction_ir.append(self.interior.party_wall_construction.U_value * self.interior.party_wall_construction.area * (self.room.temperature - self.interior.temperature))

        # convert interior pending heat into temperature differences
        self.interior.temperature += self.interior.pending_heat / self.interior.TotalHeatCapacity

        self.interior.pending_heat = 0

        # room conduction with exterior
        self.room.pending_heat += self.room.HeatExchangeCoeff * (self.exterior.weather.temperature - self.room.temperature)
        self.results.conduction_re.append(self.room.HeatExchangeCoeff * (self.exterior.weather.temperature - self.room.temperature))


        # convert room pending heat into temperature differences
        self.room.temperature += self.room.pending_heat / self.room.TotalHeatCapacity

        self.room.pending_heat = 0


        # heating and cooling for interior

        if self.interior.temperature < self.interior.heating_threshold:
            self.results.interior_heating_load.append(self.interior.TotalHeatCapacity * (self.interior.heating_goal - self.interior.temperature))
            self.interior.temperature = self.interior.heating_goal
        else:
            self.results.interior_heating_load.append(0)
        if self.interior.temperature > self.interior.cooling_threshold:
            self.results.interior_cooling_load.append(self.interior.TotalHeatCapacity * (self.interior.cooling_goal - self.interior.temperature))
            self.interior.temperature = self.interior.cooling_goal
        else:
            self.results.interior_cooling_load.append(0)

        # heating and cooling for room

        if self.room.temperature < self.room.heating_threshold:
            self.results.room_heating_load.append(self.room.TotalHeatCapacity * (self.room.heating_goal - self.room.temperature))
            self.room.temperature = self.room.heating_goal
        else:
            self.results.room_heating_load.append(0)
        if self.room.temperature > self.room.cooling_threshold:
            self.results.room_cooling_load.append(self.room.TotalHeatCapacity * (self.room.cooling_goal - self.room.temperature))
            self.room.temperature = self.room.cooling_goal
        else:
            self.results.room_cooling_load.append(0)


    def UpdateVentilationRate(self):
        #TODO
        pass


    def UpdateResult(self):
        self.results.exterior_temperature.append(self.weather.temperature)
        self.results.interior_temperature.append(self.interior.temperature)
        self.results.wintergarden_temperature.append(self.wintergarden.temperature)
        self.results.room_temperature.append(self.room.temperature)






# ====================================#
class SimulationResults:
    def __init__(self):


        self.exterior_temperature = []
        self.wintergarden_temperature = []
        self.interior_temperature = []
        self.room_temperature = []

        self.wintergarden_ventilation_rate = []

        # solar gains
        self.wintergarden_radiation_from_anisotropic = []
        self.wintergarden_radiation_from_isotropic = []

        self.interior_radiation_from_anisotropic = []
        self.interior_radiation_from_isotropic_south = []
        self.interior_radiation_from_isotropic_north = []

        self.shading_radiation_from_isotropic = []
        self.shading_radiation_from_anisotropic = []

        # ventilation and infiltration heat exchange
        self.ventilation_heat_exchange_ew = []
        self.ventilation_heat_exchange_wi = []
        self.ventilation_heat_exchange_ir = []
        self.ventilation_heat_exchange_re = []

        self.infiltration_heat_exchange_ew = []
        self.infiltration_heat_exchange_wi = []
        self.infiltration_heat_exchange_ir = []
        self.infiltration_heat_exchange_re = []


        # conduction
        self.conduction_ew = []
        self.conduction_wi = []
        self.conduction_ir = []
        self.conduction_re = []

        # energy loads
        self.interior_heating_load = []
        self.interior_cooling_load = []
        self.room_heating_load = []
        self.room_cooling_load = []

    def GroupByDailyAverage(self, result: List[float]):
        if len(result) == 8760:
            grouped = []
            for i in range(365):
                daily_result = result[i * 24: i * 24 + 24]
                daily_total = 0.0
                for hourly_result in daily_result:
                    daily_total += hourly_result
                daily_total = daily_total / 24
                grouped.append(daily_total)
            return grouped
        return -1

    def GroupByDailySummation(self, result: List[float]):
        if len(result) == 8760:
            grouped = []
            for i in range(365):
                daily_result = result[i * 24: i * 24 + 24]
                daily_total = 0.0
                for hourly_result in daily_result:
                    daily_total += hourly_result
                grouped.append(daily_total)
            return grouped
        return -1

    def Sum(self, result: List[float]):
        sum = 0
        if len(result) == 8760:
            for num in result:
                sum += num

        return sum



    def WriteToFile(self, name):
        timestamp = round(time.time())
        os.mkdir("results/" + str(timestamp) + name)
        radiation_name = "results/" + str(timestamp) + name + "/radiation.txt"
        radiation_file = open(radiation_name, "w")
        for i in range(8760):
            radiation_file.write(
                str(self.wintergarden_radiation_from_anisotropic[i]) + "," +
                str(self.wintergarden_radiation_from_isotropic[i]) + "," +
                str(self.interior_radiation_from_anisotropic[i]) + "," +
                str(self.interior_radiation_from_isotropic_south[i]) + "," +
                str(self.interior_radiation_from_isotropic_north[i]) + "," +
                str(self.shading_radiation_from_anisotropic[i]) + "," +
                str(self.shading_radiation_from_isotropic[i]) + "\n"
            )
        radiation_file.close()

        temperature_name = "results/" + str(timestamp) + name + "/temperature.txt"
        temperature_file = open(temperature_name, "w")
        for i in range(8760):
            temperature_file.write(
                str(self.exterior_temperature[i]) + "," +
                str(self.wintergarden_temperature[i]) + "," +
                str(self.interior_temperature[i]) + "," +
                str(self.room_temperature[i]) + "\n"
            )
        temperature_file.close()

        convection_name = "results/" + str(timestamp) + name + "/convection.txt"
        convection_file = open(convection_name, "w")
        for i in range(8760):
            convection_file.write(
                str(self.ventilation_heat_exchange_ew[i]) + "," +
                str(self.ventilation_heat_exchange_wi[i]) + "," +
                str(self.ventilation_heat_exchange_ir[i]) + "," +
                str(self.ventilation_heat_exchange_re[i]) + "," +
                str(self.infiltration_heat_exchange_ew[i]) + "," +
                str(self.infiltration_heat_exchange_wi[i]) + "," +
                str(self.infiltration_heat_exchange_ir[i]) + "," +
                str(self.infiltration_heat_exchange_re[i]) + "\n"
            )
        convection_file.close()

        conduction_name = "results/" + str(timestamp) + name + "/conduction.txt"
        conduction_file = open(conduction_name, "w")
        for i in range(8760):
            conduction_file.write(
                str(self.conduction_ew[i]) + "," +
                str(self.conduction_wi[i]) + "," +
                str(self.conduction_ir[i]) + "," +
                str(self.conduction_re[i]) + "\n"
            )
        conduction_file.close()

        loads_name = "results/" + str(timestamp) + name + "/loads.txt"
        loads_file = open(loads_name, "w")
        for i in range(8760):
            loads_file.write(
                str(self.interior_heating_load[i]) + "," +
                str(self.interior_cooling_load[i]) + "," +
                str(self.room_heating_load[i]) + "," +
                str(self.room_cooling_load[i]) + "\n"
            )
        loads_file.close()


        summary_name = "results/" + str(timestamp) + name + "/summary.txt"
        summary_file = open(summary_name, "w")
        grouped_temperature = self.GroupByDailyAverage(self.exterior_temperature)
        grouped_wintergarden_temperature = self.GroupByDailyAverage(self.wintergarden_temperature)
        grouped_interior_temperature = self.GroupByDailyAverage(self.interior_temperature)
        for i in range(365):
            summary_file.write(
                str(grouped_temperature[i]) + "," +
                str(grouped_wintergarden_temperature[i]) + "," +
                str(grouped_interior_temperature[i]) + "\n"
            )
        summary_file.close()

        typical_dates_name = "results/" + str(timestamp) + name + "/typical_dates.txt"
        typical_dates_file = open(typical_dates_name, "w")
        start_hours = [720, 1392, 2136, 2856, 3600, 4320, 5064, 5808, 6528, 7272, 7992, 8736]
        for start_hour in start_hours:
            for i in range(24):
                hour = start_hour + i
                typical_dates_file.write(
                    str(hour) + "," +
                    str(self.exterior_temperature[hour]) + "," +
                    str(self.wintergarden_temperature[hour]) + "," +
                    str(self.interior_temperature[hour]) + "," +
                    str(self.room_temperature[hour]) + "\n"
                )
        typical_dates_file.close()



        # 0304 TODO ADD REPRESENTATIONAL DATES











# ====================================#
LCY = open("LCY.txt", "r", encoding = "utf-8")
weatherset = WeatherSet()
for line in LCY:
    line = line.split(",")
    for i in range(len(line)):
        line[i] = eval(line[i])
    weatherset.AddRecord(
        Weather(
            line[0],
            line[1],
            line[3],
            line[2],
            line[4],
            line[5],
            line[6],
            line[7],
            line[8]
        )
    )

d = 3000

north_wall = Construction(name = "North Facade Exterior Wall", area = 44, thickness = 0.4, density = 4800, U = 3)#TEST UVALUE
north_wall.SetTestValues(0.3, 0.2, 0.8, 0)
north_aperture = Construction(name = "North Facade Window", area = 20, thickness = 0.1, density = 1200, U = 5)
north_aperture.SetTestValues(2, 0.05, 0.4, 0.55)
interior_floor = Construction(name = "Interior Floor", area = 180, thickness = 0.55, density = d, U = 0.15)
room_floor = Construction(name = "Room Floor", area = 64, thickness = 0.55, density = d, U = 0.15)
screen1 = Construction(name = "Wintergarden Glazing", area = 32, height = 8, thickness = 0.04, density = 1400, U = 2)#TTTT
screen1.SetTestValues(2, 0.05, 0.25, 0.7)
screen2 = Construction(name = "Wintergarden Glazing", area = 32, height = 8, thickness = 0.02, density = 1200, U = 2)#TTTT
screen2.SetTestValues(2, 0.05, 0.25, 0.7) #UART
shading = Construction(name = "Shading", area = 40, thickness = 0.2, density = 800)
shading.SetTestValues(2, 0.05, 0.95, 0.0)
inner_screen = Construction(name = "Interior Screen", area = 20, height = 6, thickness = 0.1, density = 2600, U = 0.8)
inner_screen.SetTestValues(2, 0.05, 0.35, 0.60)
inner_opaque = Construction(name = "Interior Opaque", area = 44, height = 6, thickness = 0.4, density = 7400, U = 0.2)
inner_opaque.SetTestValues(0.3, 0.2, 0.8, 0.0)
wintergarden_floor = Construction(name = "Wintergarden Floor", area = 16, thickness = 0.1, density = 7000, U = 1.8)
wintergarden_floor.SetTestValues(1, 0.45, 0.55, 0)
party_wall = Construction(name = "Party Wall", area = 12, thickness = 0.3, density = 1200, U = 2)




test_weather = Weather(0, 20, 120, 30, 600, 120, 3, 60, 101204)
wintergarden = Wintergarden(test_weather, [inner_screen, inner_opaque], 8, 1.60)
wintergarden.maximum_opening = 12




wintergarden.SetFloorConstruction(wintergarden_floor)
wintergarden.SetShadingConstruction(shading)
exterior = Exterior(test_weather, [screen1, screen2])
interior = Interior(test_weather, [], 16, 10.0)
interior.SetFloorConstruction(interior_floor)
interior.SetPartyWallConstruction(party_wall)
room = Room(test_weather, [north_aperture, north_wall])
room.SetPartyWallConstruction(party_wall)
room.SetFloorConstruction(room_floor)


strategy = Strategy()
strategy.shading_ratio = 0.0
strategy.opening_ratio = 0
strategy.interior_opening_ratio = 0
strategy.north_opening_ratio = 0
strategy.heating_threshold = 14
strategy.heating_goal = 15
strategy.cooling_threshold = 28
strategy.cooling_goal = 28
strategy.DoSummerShading(True)
strategy.DoEncloseWintergarden(True)
strategy.DoSummerVentilation(True)

strategy.summer_window_u_value_multiplier = 5

strategyset = StrategySet()
strategyset.AddStrategy(strategy)



test_model = Model(exterior, wintergarden, interior, room, weatherset, strategyset)
test_model.UpdateWeather()
test_model.UpdateModel()
while test_model.ready:
    test_model.UpdateModel()

print(str(test_model.results.Sum(test_model.results.interior_heating_load) / 3600) + " kWh")
print(str(test_model.results.Sum(test_model.results.interior_cooling_load) / 3600) + " kWh")
print(str(test_model.results.Sum(test_model.results.room_heating_load) / 3600) + " kWh")
print(str(test_model.results.Sum(test_model.results.room_cooling_load) / 3600) + " kWh")

#test_model.results.WriteToFile("_0309_enc_v_s")

plt.plot(test_model.results.interior_temperature)
plt.plot(test_model.results.room_temperature)
plt.plot(test_model.results.exterior_temperature)
plt.plot(test_model.results.wintergarden_temperature)
plt.show()

'''
p1 = ProceduralRadiance(600, 100, 0.3, 30, 120, Vector3d(0, -1, 0))
p1.Run()
print (p1.g_dir)
p2 = ProceduralRadiance(200, 100, 0.3, 30, 120, Vector3d(0, -1, 0))
p2.Run()
print (p2.g_dir)
'''
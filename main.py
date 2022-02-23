#from Rhino.Geometry import *
from typing import List
import math
import time
import os


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

        #GLOBAL OVERRIDES INITIALIZATION FOR INTERNAL TESTS ONLY
        self.heat_capacity = 2000
        self.U_value = U
        self.absorbance = 0.3
        self.reflectivity = 0.7
        self.transmittance = 0.0


    def AddLayer(self, material: OpaqueConstructionMaterial):
        self.materials.append(material)
        self.thickness += material.thickness
        # TODO
        # MASS

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

    @property
    def Mass(self):
        return self.thickness * self.area * self.density

    @property
    def HeatExchangeCoeff(self):
        return self.U_value * self.area

    @property
    def HeatCapacity(self):
        return self.heat_capacity

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
        return 1005.0

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



class Strategy:
    def __init__(self, opening_ratio=1.0, shading_ratio=0.05, expose_interior=False, heating_threshold = 10.0, cooling_threshold = 35.0, internal_gain = 1200):
        self.opening_ratio = opening_ratio
        self.shading_ratio = shading_ratio
        self.expose_interior = expose_interior
        self.heating_threshold = heating_threshold
        self.cooling_threshold = cooling_threshold
        self.interior_internal_gain = internal_gain

    def __str__(self):
        # TODO: describe different strategies in strings
        pass

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
            coeff += construction.area * construction.HeatExchangeCoeff
        return coeff

class Wintergarden:
    def __init__(self, weather: Weather, constructions: List[Construction], maximum_opening, offset):
        self.weather = weather
        self.maximum_opening = maximum_opening
        self.offset = offset

        self.opening_ratio = 0.8
        self.opening = self.opening_ratio * self.maximum_opening
        self.constructions = constructions

        self.area = 0
        self.height = 8.0
        for construction in self.constructions:
            self.area += construction.area

        self.pending_heat = 0.0

        self.temperature = 15.0
        self.exterior_ventilation_rate = 0.0
        self.exterior_infiltration_rate = 0.0
        self.interior_ventilation_rate = 0.0
        self.interior_infiltration_rate = 0.0

    def ApplyStrategy(self, strategy: Strategy):
        self.opening_ratio = strategy.opening_ratio
        self.opening = self.opening_ratio * self.maximum_opening

    def SetFloorConstruction(self, construction: Construction):
        self.floor_construction = construction

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
            coeff += construction.area * construction.HeatExchangeCoeff
        return coeff

    @property
    def HeatCapacity(self):
        hc = 0.0
        for construction in self.constructions:
            hc += construction.Mass * construction.HeatCapacity
        hc += self.area * self.height * self.weather.GetAirDensity()
        return hc

class Interior:
    def __init__(self, weather: Weather, constructions: List[Construction]):
        self.weather = weather

        self.shading_ratio = 0.05
        self.expose_interior = False

        self.constructions = constructions

        self.area = 0.0
        self.height = 6.5

        for construction in self.constructions:
            self.area += construction.area

        self.pending_heat = 0.0

        self.temperature = 15.0
        self.exterior_ventilation_rate = 0.0
        self.wintergarden_ventilation_rate = 0.0


        self.heating_threshold = 10.0
        self.cooling_threshold = 35.0
        self.internal_gain = 0.0

    def ApplyStrategy(self, strategy: Strategy):
        self.shading_ratio = strategy.shading_ratio
        self.expose_interior = strategy.expose_interior
        self.expose_interior = strategy.expose_interior
        self.internal_gain = strategy.interior_internal_gain
        self.heating_threshold = strategy.heating_threshold
        self.cooling_threshold = strategy.cooling_threshold


    def SetFloorConstruction(self, construction: Construction):
        self.floor_construction = construction

    @property
    def Mass(self):
        mass = 0.0
        for construction in self.constructions:
            mass += construction.Mass
        mass += self.floor_construction.Mass
        return mass

    @property
    def HeatExchangeCoeff(self):
        coeff = 0.0
        for construction in self.constructions:
            coeff += construction.area * construction.HeatExchangeCoeff
        return coeff

    @property
    def HeatCapacity(self):
        hc = 0.0
        for construction in self.constructions:
            hc += construction.Mass * construction.HeatCapacity
        hc += self.area * self.height * self.weather.GetAirDensity()
        return hc

class Model:
    def __init__(self, exterior: Exterior, wintergarden: Wintergarden, interior: Interior, weatherset: WeatherSet, strategyset: StrategySet, starting_hour = 0, analysis_period = -1):
        self.exterior = exterior
        self.wintergarden = wintergarden
        self.interior = interior
        self.weatherset = weatherset
        self.weather = weatherset.GetRecord(0)
        self.strategyset = strategyset
        self.strategy = strategyset.GetStrategy(0)
        self.hour = starting_hour
        self.analysis_period = analysis_period
        self.ready = True
        self.results = SimulationResults()

    def UpdateModel(self):
        print(self.hour)
        self.UpdateStrategy()
        self.UpdateWeather()
        self.UpdateSolarGain()
        self.UpdateVentilationLoss()
        self.UpdateLatentHeat()
        self.UpdateInternalGain()
        self.UpdateHeatTransfer()
        self.UpdateVentilationRate()
        self.UpdateResult()

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
        # TEST
        self.strategy = self.strategyset.GetStrategy(0)#self.hour#)
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

        for construction in self.wintergarden.constructions: # second screen layer
            self.procedural_radiance_object.normal = construction.normal
            self.procedural_radiance_object.Run()

            local_anisotropic_radiation = inner_screen_anisotropic_radiation * construction.area / self.wintergarden.area
            local_isotropic_radiation = inner_screen_isotropic_radiation * construction.area / self.wintergarden.area

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


        for construction in self.interior.constructions: # north facing facade
            self.procedural_radiance_object.normal = construction.normal
            self.procedural_radiance_object.Run()

            iso_rad = self.procedural_radiance_object.GetIsotropicRadiation() * construction.area

            # isotropic radiation transmitted into the interior
            interior_absorbed_radiation += iso_rad * construction.GetAverageTransmittance()
            # update to simulation result energy breakdowns
            interior_radiation_from_isotropic_north += iso_rad * construction.GetAverageTransmittance()

            # isotropic radiation absorbed by the fabric
            interior_absorbed_radiation += iso_rad * construction.GetAverageAbsorbance()
            # update to simulation result energy breakdowns
            interior_radiation_from_isotropic_north += iso_rad * construction.GetAverageAbsorbance()


        self.wintergarden.pending_heat += wintergarden_absorbed_radiation
        self.interior.pending_heat += interior_absorbed_radiation

        self.results.wintergarden_radiation_from_anisotropic.append(wintergarden_radiation_from_anisotropic)
        self.results.wintergarden_radiation_from_isotropic.append(wintergarden_radiation_from_isotropic)
        self.results.interior_radiation_from_isotropic_north.append(interior_radiation_from_isotropic_north)
        self.results.interior_radiation_from_isotropic_south.append(interior_radiation_from_isotropic_south)
        self.results.interior_radiation_from_anisotropic.append(interior_radiation_from_anisotropic)



        # TEST
        # print(wintergarden_absorbed_radiation)
        # print(interior_absorbed_radiation)


    def UpdateVentilationLoss(self):

        # heat exchange in forms of VENTILATION, and INFILTRATION between the exterior and the wintergarden, between the wintergarden and the interior
        self.wintergarden.pending_heat += self.weather.GetExteriorAirHeatCapacity() * (- self.wintergarden.temperature + self.exterior.weather.temperature) * (self.wintergarden.exterior_ventilation_rate + self.wintergarden.exterior_infiltration_rate)
        self.wintergarden.pending_heat += self.weather.GetExteriorAirHeatCapacity() * (- self.wintergarden.temperature + self.interior.temperature) * (self.wintergarden.interior_ventilation_rate + self.wintergarden.interior_infiltration_rate)
        self.interior.pending_heat += self.weather.GetExteriorAirHeatCapacity() * (self.wintergarden.temperature - self.interior.temperature) * (self.wintergarden.interior_ventilation_rate + self.wintergarden.interior_infiltration_rate)

        # TODO
        # CONSIDER LATENT HEAT GAIN / PSYCHROMETRIC PROCESSES

    def UpdateLatentHeat(self):
        # TODO
        pass

    def UpdateInternalGain(self):
        self.interior.pending_heat += self.strategy.interior_internal_gain


    def UpdateHeatTransfer(self):
        # conduction with exterior
        self.wintergarden.pending_heat += self.exterior.HeatExchangeCoeff * (self.exterior.weather.temperature - self.wintergarden.temperature)

        # conduction with interior
        self.wintergarden.pending_heat += self.wintergarden.HeatExchangeCoeff * (self.interior.temperature - self.wintergarden.temperature)

        # convert pending heat into temperature differences

        self.wintergarden.temperature += self.wintergarden.pending_heat / self.wintergarden.HeatCapacity / self.wintergarden.Mass
        self.wintergarden.pending_heat = 0

        print(wintergarden.temperature)

        # TODO
        # add conduction heat exchange between other environments involved


    def UpdateVentilationRate(self):
        #TODO
        pass


    def UpdateResult(self):
        self.results.exterior_temperature.append(self.weather.temperature)
        self.results.interior_temperature.append(self.interior.temperature)
        self.results.wintergarden_temperature.append(self.wintergarden.temperature)






# ====================================#
class SimulationResults:
    def __init__(self):

        self.exterior_temperature = []
        self.wintergarden_temperature = []
        self.interior_temperature = []

        self.wintergarden_ventilation_rate = []

        self.wintergarden_radiation_from_anisotropic = []
        self.wintergarden_radiation_from_isotropic = []

        self.interior_radiation_from_anisotropic = []
        self.interior_radiation_from_isotropic_south = []
        self.interior_radiation_from_isotropic_north = []






        self.heating_load = []
        self.cooling_load = []
        self.passive_heating = []
        self.passive_cooling = []
        self.passive_ventilation = []

    def GroupByDailyAverage(self, result: List[float]):
        if len(result) == 8760:
            grouped = []
            for i in range(365):
                daily_result = result[i: i + 24]
                daily_total = 0.0
                for hourly_result in daily_result:
                    daily_total += hourly_result
                daily_total = daily_total / 24
                grouped.append(daily_total)
            return grouped
        return -1

    def WriteToFile(self):
        timestamp = time.time() * 100
        os.mkdir(str(timestamp))
        radiation_name = str(timestamp) + "/radiation.txt"
        radiation_file = open(radiation_name, "w")
        for i in range(8760):
            radiation_file.write(
                str(self.wintergarden_radiation_from_anisotropic[i]) + "," +
                str(self.wintergarden_radiation_from_isotropic[i]) + "," +
                str(self.interior_radiation_from_anisotropic[i]) + "," +
                str(self.interior_radiation_from_isotropic_south[i]) + "," +
                str(self.interior_radiation_from_isotropic_north[i]) + "\n"
            )
        radiation_file.close()
        """
        summary_name = str(timestamp) + "/summary.txt"
        summary_file = open(summary_name, "w")

        summary_file.close()
        """













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
print(weatherset.Length)



north_wall = Construction(name = "North Facade Exterior Wall", area = 5, thickness = 0.4, density = 2.4, U = 0.3)
north_aperture = Construction(name = "North Facade Window", area = 3, thickness = 0.02, density = 1.2, U = 1.2)
interior_floor = Construction(name = "Interior Floor", area = 90, thickness = 0.45, density = 2.4, U = 0.15)
screen1 = Construction(name = "Wintergarden Glazing", area = 32, height = 8, thickness = 0.04, density = 1.4, U = 2.4)
screen1.SetTestValues(1, 0.1, 0.1, 0.8)
screen2 = Construction(name = "Wintergarden Glazing", area = 32, height = 8, thickness = 0.02, density = 1.2, U = 3)
screen2.SetTestValues(1, 0.3, 0.3, 0.4) #UART
inner_screen = Construction(name = "Interior Screen", area = 48, height = 6, thickness = 0.1, density = 1.6, U = 1.0)
inner_screen.SetTestValues(1, 0, 0.9, 0.1)
wintergarden_floor = Construction(name = "Wintergarden Floor", area = 1.2 * 8, thickness = 0.3, density = 1.8, U = 1.8)
wintergarden_floor.SetTestValues(1, 0.3, 0.7, 0)


test_weather = Weather(0, 20, 120, 30, 600, 120, 3, 60, 101204)
wintergarden = Wintergarden(test_weather, [inner_screen], 2, 1.2)

wintergarden.SetFloorConstruction(wintergarden_floor)
exterior = Exterior(test_weather, [screen1, screen2])
interior = Interior(test_weather, [north_wall])
interior.SetFloorConstruction(interior_floor)



strategy = Strategy()
strategyset = StrategySet()
strategyset.AddStrategy(strategy)


test_model = Model(exterior, wintergarden, interior, weatherset, strategyset)
test_model.UpdateWeather()
test_model.UpdateModel()
while test_model.ready:
    test_model.UpdateModel()
test_model.results.WriteToFile()
'''
p1 = ProceduralRadiance(600, 100, 0.3, 30, 120, Vector3d(0, -1, 0))
p1.Run()
print (p1.g_dir)
p2 = ProceduralRadiance(200, 100, 0.3, 30, 120, Vector3d(0, -1, 0))
p2.Run()
print (p2.g_dir)
'''
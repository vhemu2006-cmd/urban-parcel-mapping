def generate_land_use_report(land_use_analysis):
    """
    AI-style analysis agent.
    Takes land-use percentages and generates
    a human-readable urban analysis report.
    """

    # Extract percentages
    background = land_use_analysis.get(
        "Background", {}
    ).get("percentage", 0)

    building = land_use_analysis.get(
        "Building", {}
    ).get("percentage", 0)

    road = land_use_analysis.get(
        "Road", {}
    ).get("percentage", 0)

    water = land_use_analysis.get(
        "Water", {}
    ).get("percentage", 0)

    barren = land_use_analysis.get(
        "Barren Land", {}
    ).get("percentage", 0)

    forest = land_use_analysis.get(
        "Forest", {}
    ).get("percentage", 0)

    agriculture = land_use_analysis.get(
        "Agriculture", {}
    ).get("percentage", 0)


    # -------------------------------------------------
    # Determine dominant land-use category
    # -------------------------------------------------

    categories = {
        "Background": background,
        "Building": building,
        "Road": road,
        "Water": water,
        "Barren Land": barren,
        "Forest": forest,
        "Agriculture": agriculture
    }

    dominant_category = max(
        categories,
        key=categories.get
    )

    dominant_percentage = categories[
        dominant_category
    ]


    # -------------------------------------------------
    # Generate general observation
    # -------------------------------------------------

    observations = []

    if building > 30:
        observations.append(
            "The region has a high concentration of built-up structures."
        )
    elif building > 15:
        observations.append(
            "The region contains a moderate amount of built-up area."
        )
    else:
        observations.append(
            "The built-up area is relatively limited."
        )


    if forest > 20:
        observations.append(
            "The region has substantial vegetation coverage."
        )
    elif forest > 5:
        observations.append(
            "The region contains some green vegetation areas."
        )
    else:
        observations.append(
            "Vegetation coverage is relatively low."
        )


    if water > 10:
        observations.append(
            "Significant water bodies are present in the region."
        )
    elif water > 1:
        observations.append(
            "Small water bodies may be present."
        )


    if road > 10:
        observations.append(
            "The region has considerable road connectivity."
        )
    elif road > 3:
        observations.append(
            "The region contains some road infrastructure."
        )
    else:
        observations.append(
            "Road coverage is relatively low."
        )


    if barren > 10:
        observations.append(
            "A considerable amount of barren land is available."
        )


    if agriculture > 10:
        observations.append(
            "Agricultural land occupies a notable portion of the region."
        )


    # -------------------------------------------------
    # Generate recommendations
    # -------------------------------------------------

    recommendations = []

    if building > 30 and forest < 10:
        recommendations.append(
            "Consider increasing green spaces and tree plantation "
            "in highly built-up areas."
        )

    if water > 10:
        recommendations.append(
            "Water bodies should be protected from encroachment "
            "and pollution."
        )

    if road < 3 and building > 20:
        recommendations.append(
            "Road connectivity and transportation infrastructure "
            "may require further assessment."
        )

    if forest > 15:
        recommendations.append(
            "Existing vegetation should be preserved during "
            "future urban development."
        )

    if barren > 10:
        recommendations.append(
            "Barren regions may be evaluated for suitable "
            "planned development or ecological restoration."
        )

    if not recommendations:
        recommendations.append(
            "The region shows a mixed land-use pattern. "
            "Further geographic and planning data may be useful "
            "for detailed assessment."
        )


    # -------------------------------------------------
    # Final report
    # -------------------------------------------------

    report = {
        "dominant_land_use": dominant_category,
        "dominant_percentage": dominant_percentage,

        "summary": (
            f"The dominant land-use category is "
            f"{dominant_category}, covering approximately "
            f"{dominant_percentage}% of the analyzed image."
        ),

        "observations": observations,

        "recommendations": recommendations
    }

    return report
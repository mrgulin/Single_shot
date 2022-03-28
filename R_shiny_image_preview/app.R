#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Old Faithful Geyser Data"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
            sliderInput("n_e",
                        "number of electrons:",
                        min = 2,
                        max = 10,
                        value = 6, step = 2),
            sliderInput("u",
                        "U parameter:",
                        min = 2,
                        max = 10,
                        value = 4, step = 2),
            radioButtons('plotname', 'Which plot to plot', c('energy'='Energy_errors.png', 'density'='Densities.png'), )
        ),

        # Show a plot of the generated distribution
        mainPanel(
          textOutput("showInput"),
          fluidRow(imageOutput('imgplot')),
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
    vals <- reactiveValues()
    
    observe(
      vals$path_plot <- paste("chain1_ns-6_ne-", input$n_e, "_u-", input$u , "_dist-0.1/", input$plotname, sep=''))

    
    
    output$imgplot <- renderImage({
      # When input$n is 1, filename is www/images/image1.jpeg
      filename <- normalizePath(file.path('C:/Users/tinc9/Documents/CNRS-offline/quantum_main_project/LPFET/results', vals$path_plot
                                          , sep=''))
      
      # Return a list containing the filename
      list(src = filename, width = 1000)
    }, deleteFile = FALSE)
}

# Run the application 
shinyApp(ui = ui, server = server)
